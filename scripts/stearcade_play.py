"""Dreamer playtest in native STE Hatari; private instrumented builds, real RAM.

Run with the bundled native CPython. Local STEArcade and F030Arcade-private
checkouts provide the port and the established Windows named-pipe transport.
This is a bounded exploration test, never a proof of complete playability.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
STE = Path('C:/Arbeit/STEArcade')
ARC = Path('C:/Arbeit/F030Arcade-private')
os.environ['PATH'] = 'C:/msys64/ucrt64/bin;C:/msys64/usr/bin;' + os.environ.get('PATH', '')
sys.path[:0] = [str(ROOT), str(STE / 'engine'), str(ARC / 'scripts')]
import numpy as np
import torch
from dreamerv3.agent import Agent
from dreamerv3.replay import Replay
from dreamerv3.train import parser, seed_all
from exploration.hatari_live import Hatari, Pipe, Closed
from ppo_profiles import profile, boot_inputs, actions
from paths import VASM, VLINK, RT_INCLUDE

HATARI = ARC / 'third_party/hatari/build-ucrt64-merge/src/hatari.exe'
TOS = ARC / 'third_party/tos/tos206de.img'


def capture_commands(out, sym, game, label):
    ptr = sym['display_screen_address' if game == 'dkong' else 'flip_addr']
    pal = sym['dk_palette' if game == 'dkong' else 'shown_pal']
    return [f"savebin {(out / (label+'.bin')).as_posix()} '(TEXT+${ptr:x}).l' $10000",
            f"savebin {(out / (label+'.pal')).as_posix()} 'TEXT+${pal:x}' $20"]


def render_capture(raw, palette, output):
    from PIL import Image
    words = np.frombuffer(raw, dtype='>u2').reshape(256,32,4)
    pixels = np.zeros((256,512),np.uint8)
    for k in range(4):
        pixels |= (((words[:,:,k,None] >> np.arange(15,-1,-1)) & 1).reshape(256,512).astype(np.uint8) << k)
    colors = [[(((int(v)>>s)&7)*2+((int(v)>>(s+3))&1))*17 for s in [8,4,0]]
              for v in np.frombuffer(palette,dtype='>u2')]
    Image.fromarray(np.array(colors,dtype=np.uint8)[pixels[:,:320]]).resize((640,512)).save(output)


def replay_run(args, out, program, sym):
    source = args.replay.resolve()
    manifest = json.loads((source / 'manifest.json').read_text())
    discovery = json.loads((source / 'summary.json').read_text())
    if manifest['program_sha256'] != sha(program / 'GAME.PRG'):
        raise ValueError('Replay program hash differs from discovery')
    with gzip.open(source / 'states.jsonl.gz', 'rt') as f:
        expected = {row['tick']:bytes.fromhex(row['ram']) for row in map(json.loads,f)}
    h = Live(out / 'hatari', program, sym, args.game)
    checked = 0; ticks = 0; mismatch = None; failure = None
    try:
        h.step()
        with gzip.open(source / 'timeline.jsonl.gz', 'rt') as f:
            for tick, system, player, extra in map(json.loads,f):
                # Discovery screenshot paths are diagnostic side effects, not game inputs.
                extra = [c for c in extra if not c.startswith(('screenshot ', 'savebin '))]
                capture = h.step([f"w 'TEXT+${sym['dream_system']:x}' ${system:x}",
                                  f"w 'TEXT+${sym['dream_player']:x}' ${player:x}", *extra])
                ticks = tick+1
                if capture[1] != bytes([system,player]):
                    raise RuntimeError(f'Replay input mismatch at tick {ticks}')
                if ticks in expected:
                    checked += 1
                    if capture[0] != expected[ticks]:
                        offsets=[i for i,(a,b) in enumerate(zip(expected[ticks],capture[0])) if a!=b]
                        mismatch=dict(tick=ticks, differing_bytes=len(offsets), first_offsets=offsets[:16])
                        (out/'expected.bin').write_bytes(expected[ticks]);(out/'observed.bin').write_bytes(capture[0])
                        break
    except Exception as exc:
        failure = repr(exc)
    finally:
        h.close()
    crash = program / 'CRASH.TXT'
    crash_text = crash.read_text() if crash.exists() else None
    report=dict(ticks=ticks,expected_ticks=discovery['ticks'],checked=checked,mismatch=mismatch,
                failure=failure, crash=crash_text,
                passed=failure is None and mismatch is None and checked==len(expected) and ticks==discovery['ticks'])
    original_crash = source / 'program/CRASH.TXT'
    report['same_crash_report'] = bool(crash_text and original_crash.exists() and crash_text==original_crash.read_text())
    (out/'replay.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    return not report['passed']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(game, out):
    src = STE / 'games' / game / 'src'
    work = out / 'program'
    work.mkdir(parents=True)
    for f in src.iterdir():
        if f.is_file(): shutil.copy2(f, work / f.name)
    main = work / f'{game}.s'
    text = main.read_text()
    old = '\t; Z80 NMI.\n' if game == 'dkong' else '.deliver:\n'
    assert text.count(old) == 1
    text = text.replace(old, old + '\tbsr dream_gate\n', 1)
    # VBLs may occur during a slow tick: the private held bytes own inputs.
    text = re.sub(r'^\s*jp_merge[^\n]*$', '; private Dreamer controls', text, flags=re.M)
    target = ('z80_block+$7f00', 'z80_block+$7e00') if game == 'dkong' else ('Z_SYSTEM', 'Z_SYSTEM+1')
    text += ('\n\tsection text,code\ndream_gate:\n'
             '\tmove.b dream_system,' + target[0] + '\n'
             '\tmove.b dream_player,' + target[1] + '\n'
             '\trts\n\tsection data,data\ndream_system: dc.b ' + ('0' if game == 'dkong' else '255') +
             '\ndream_player: dc.b ' + ('0' if game == 'dkong' else '255') + '\n\teven\n')
    main.write_text(text)
    obj, prg, mp = work / 'game.o', work / 'GAME.PRG', work / 'game.map'
    with (out / 'build.log').open('w') as log:
        for cmd in ([VASM, '-Fvobj', '-quiet', '-nowarn=2047', '-I', str(STE / 'build' / game),
                     '-I', str(RT_INCLUDE), '-o', str(obj), main.name],
                    [VLINK, '-b', 'ataritos', '-s', '-o', str(prg), '-M' + str(mp), str(obj)]):
            subprocess.run(cmd, cwd=work, stdout=log, stderr=log, check=True)
    symbols = {name: int(addr, 16) for addr, name in re.findall(r'^\s*0x([0-9a-f]+) (\w+):', mp.read_text(), re.M)}
    return work, symbols


class Live(Hatari):
    """Same tested Pipe/watchdog transport; TEXT-relative PRG observation addresses."""
    def __init__(self, out, program, sym, game):
        self.out, self.timeout = out, 45
        out.mkdir()
        self.pipe = Pipe(f'ste-dreamer-{os.getpid()}')
        self.obs = [out / 'ram.bin', out / 'input.bin']
        base = sym['z80_block'] + 0x6000 if game == 'dkong' else sym['g_block'] + 0x6000
        self.base = base
        size = 0xc00 if game == 'dkong' else 0x1000
        self.stepfile = out / 'step.ini'
        self.stepfile.write_text(f"savebin {self.obs[0].as_posix()} 'TEXT+${base:x}' ${size:x}\n"
                                 f"savebin {self.obs[1].as_posix()} 'TEXT+${sym['dream_system']:x}' $2\n"
                                 f"parse {self.pipe.guest_path}\n")
        entry = out / 'entry.ini'
        arm = out / 'arm.ini'
        arm.write_text(f"b pc = 'TEXT+${sym['dream_gate']:x}' :trace :quiet :file {self.stepfile.as_posix()}\n")
        entry.write_text(f"b VBL = 'VBL+1' :once :trace :quiet :file {arm.as_posix()}\n")
        env = dict(os.environ, PATH='C:/msys64/ucrt64/bin;' + os.environ.get('PATH', ''))
        self.argv = [str(HATARI), '--machine', 'ste', '--memsize', '4', '--tos', str(TOS),
                     '--fast-boot', 'true', '--cpu-exact', 'true', '--compatible', 'true',
                     '--sound', 'off', '--confirm-quit', 'off', '--statusbar', 'false',
                     '--fast-forward', 'true', '--disable-video', 'true', '--frameskips', '0',
                     '--parse', 'prg:' + str(entry), '--log-file', str(out / 'hatari.log'), str(program / 'GAME.PRG')]
        self.log = (out / 'process.log').open('wb')
        self.proc = subprocess.Popen(self.argv, cwd=program, env=env, stdout=self.log, stderr=self.log,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
        self.last, self.ticks, self.pending = time.monotonic(), 0, None
        threading.Thread(target=self._watch, daemon=True).start()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('game', choices=['dkong', '1942'])
    ap.add_argument('out', type=Path)
    ap.add_argument('--decisions', type=int, default=1800)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--assist', action='store_true', help='refill lives; 1942 invincibility and periodic finish cheat')
    ap.add_argument('--replay', type=Path, help='replay an existing completed run and compare every saved RAM checkpoint')
    args = ap.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    seed_all(args.seed); torch.set_num_threads(2)
    p = profile(args.game); macros = actions(p)
    program, sym = build(args.game, out)
    manifest = dict(args=vars(args), hatari=str(HATARI), hatari_sha256=sha(HATARI),
                    tos_sha256=sha(TOS), program_sha256=sha(program / 'GAME.PRG'),
                    ste_revision=subprocess.check_output(['git', '-C', str(STE), 'rev-parse', 'HEAD'], text=True).strip(),
                    private_source_sha256=sha(program / f'{args.game}.s'), profile=p,
                    note='Instrumented timing/input path; sound disabled; no full-game or visual-equivalence claim.')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
    if args.replay:
        return replay_run(args,out,program,sym)
    cfg = vars(parser().parse_args([])); cfg.update(device='cuda', hidden=128, deter=128, envs=1, batch=8, length=16)
    obs_dim = 512
    agent = Agent(obs_dim, len(macros), cfg)
    buf = Replay(10000, 1, obs_dim, len(macros), cfg['deter'], cfg['latents'] * cfg['classes'], args.seed)
    h = Live(out / 'hatari', program, sym, args.game)
    timeline = gzip.open(out / 'timeline.jsonl.gz', 'wt')
    states = gzip.open(out / 'states.jsonl.gz', 'wt')
    events = (out / 'events.jsonl').open('w')
    t0 = time.monotonic(); last = t0
    tick = 0; last_ram = None; frozen = 0; updates = 0
    stage_seen, lives_seen = set(), set(); anomalies = []; transitions = []
    ram = None; decisions = 0; started = False
    def event(kind, **kw):
        row = dict(kind=kind, tick=tick, seconds=round(time.monotonic()-t0, 1), **kw)
        events.write(json.dumps(row)+'\n'); events.flush(); print(json.dumps(row), flush=True)
    def frame(system, player, extra=()):
        nonlocal tick, ram, last_ram, frozen
        commands = [f"w 'TEXT+${sym['dream_system']:x}' ${system:x}",
                    f"w 'TEXT+${sym['dream_player']:x}' ${player:x}", *extra]
        timeline.write(json.dumps([tick, system, player, list(extra)])+'\n')
        capture = h.step(commands)
        if capture[1] != bytes([system, player]):
            raise RuntimeError(f'Input transport mismatch at tick {tick}: {capture[1].hex()} != {system:02x}{player:02x}')
        ram = capture[0]; tick += 1
        frozen = frozen + 1 if ram == last_ram else 0; last_ram = ram
        if frozen == 600:
            anomalies.append(dict(kind='unchanged_ram', tick=tick)); event('unchanged_ram')
        if tick % 120 == 0: states.write(json.dumps(dict(tick=tick, ram=ram.hex()))+'\n')
        return ram
    def observation():
        return np.frombuffer(ram, np.uint8)[np.linspace(0, len(ram)-1, obs_dim, dtype=int)].astype(np.float32)[None] / 255
    def values():
        return (ram[0x227], ram[0x228]) if args.game == 'dkong' else (ram[0x103], ram[0x101])
    try:
        # Initial stop precedes the first input application.
        ram = h.step()[0]
        for t in range(p['boot']): frame(*boot_inputs(p, t))
        event('boot_complete', stage=values()[0], lives=values()[1], ram_head=ram[:32].hex())
        state = agent.world.initial(1, cfg['device'])
        prev = np.zeros((1, len(macros)), np.float32); rew = np.zeros(1, np.float32); first = np.ones(1, np.float32)
        rng = np.random.default_rng(args.seed)
        for decision in range(args.decisions):
            obs = observation()
            buf.add(obs=obs, action=prev, reward=rew, first=first, terminal=np.zeros(1),
                    h=state[0].cpu().numpy(), z=state[1].cpu().numpy())
            with torch.no_grad():
                action, state = agent.act(torch.as_tensor(obs, device=cfg['device']), torch.as_tensor(prev, device=cfg['device']),
                                          torch.as_tensor(first, device=cfg['device']), state)
            a = int(rng.integers(len(macros))) if decision < 128 else int(action.argmax(-1).item())
            before = ram; oldstage, oldlives = values()
            for player in macros[a]:
                extra = []
                if tick % 3000 == 0:
                    extra.extend(capture_commands(out,sym,args.game,f'screen-{tick:06d}'))
                if args.assist:
                    lifeoffset = 0x228 if args.game == 'dkong' else 0x101
                    extra.append(f"w 'TEXT+${h.base+lifeoffset:x}' $6")
                    if args.game == 'dkong' and tick > 1800 and tick % 4800 == 0:
                        target = 1 + (tick // 4800) % 4
                        extra.append(f"w 'TEXT+${h.base+0x227:x}' ${target:x}")
                        event('assisted_stage_request', target=target)
                    if args.game == '1942':
                        extra.append(f"w 'TEXT+${h.base+0xa5:x}' $2")
                        if tick > 1600 and tick % 1200 == 0:
                            extra += [f"w 'TEXT+${h.base+off:x}' ${val:x}" for off,val in [(0xbd0,7),(0x99,1),(0x102,0x43)]]
                            event('assisted_finish_request')
                # Cabinet restart attempts are explicit, not counted as completion.
                system = p['system_idle']
                phase = (tick-p['boot']) % 6000
                if phase in range(5600,5604): system = system | p['coin_mask'] if p['high_system'] else system & ~p['coin_mask']
                if phase in range(5650,5656): system = system | p['start_mask'] if p['high_system'] else system & ~p['start_mask']
                frame(system, player, extra)
            stage, lives = values(); stage_seen.add(stage); lives_seen.add(lives)
            if stage != oldstage:
                transitions.append([tick,oldstage,stage]); event('stage_byte_change', old=oldstage, new=stage)
            if 0 < lives < 8: started = True
            novelty = np.count_nonzero(np.frombuffer(ram,np.uint8) != np.frombuffer(before,np.uint8)) / len(ram)
            rew[:] = novelty - float(0 < oldlives < 8 and lives < oldlives)
            prev = np.eye(len(macros), dtype=np.float32)[[a]]; first[:] = 0
            if decision >= 64 and decision % 4 == 0:
                batch, indices = buf.sample(cfg['batch'], cfg['length'], cfg['device'])
                metrics, hs, zs = agent.update(batch); buf.refresh(indices, hs, zs); updates += 1
            decisions = decision+1
            if time.monotonic()-last > 40:
                event('progress', decisions=decisions, updates=updates, stage=stage, lives=lives); last=time.monotonic()
        event('completed_budget', decisions=decisions, updates=updates)
    except Exception as exc:
        anomalies.append(dict(kind='exception', tick=tick, message=repr(exc)))
        event('failure', message=repr(exc))
    finally:
        h.close(); timeline.close(); states.close(); events.close()
        for raw in out.glob('screen-*.bin'):
            render_capture(raw.read_bytes(),raw.with_suffix('.pal').read_bytes(),raw.with_suffix('.png'))
        torch.save(dict(agent=agent.state_dict(), config=cfg), out / 'agent.pt')
        crash = program / 'CRASH.TXT'
        summary = dict(game=args.game, assisted=args.assist, ticks=tick, decisions=decisions, updates=updates,
                       seconds=round(time.monotonic()-t0,1), stage_bytes=sorted(stage_seen), lives_bytes=sorted(lives_seen),
                       gameplay_lives_observed=started, transitions=transitions, anomalies=anomalies,
                       crash=crash.read_text() if crash.exists() else None,
                       complete_playability_verified=False, sound_tested=False, render_equivalence_tested=False)
        (out / 'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary),flush=True)
    return bool(anomalies)


if __name__ == '__main__':
    raise SystemExit(main())
