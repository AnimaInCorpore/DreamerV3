# STEArcade Dreamer playtesting

`scripts/stearcade_play.py` runs a fresh per-game Dreamer agent directly against
native Windows Hatari. It currently has explicit adapters for `dkong` and `1942`.
It reuses STEArcade's `engine/ppo_profiles.py` input/RAM contracts and the
F030Arcade-private `scripts/exploration/hatari_live.py` named-pipe transport.

## Run

Prerequisites on this machine: native Dreamer CPython with Torch, NumPy and
Pillow; the STEArcade generated translation/assets; the local VASM/VLINK tools;
and F030Arcade-private's patched `build-ucrt64-merge` Hatari and TOS 2.06 image.
The script prepends the MSYS2 runtime DLL paths. It copies port source into its
own output directory and assembles there, preserving canonical PRGs, maps,
recordings, coverage and runtime source. Preparing missing port assets uses the
port's normal build workflow (Donkey Kong additionally needs SciPy and local
MAME sound capture).

```powershell
& C:\msys64\opt\dreamerv3-python\python.exe scripts/stearcade_play.py dkong runs/dkong-play --decisions 1800
& C:\msys64\opt\dreamerv3-python\python.exe scripts/stearcade_play.py 1942 runs/1942-play --decisions 1800
# Explicitly assisted exploration; never count these as natural clears:
& C:\msys64\opt\dreamerv3-python\python.exe scripts/stearcade_play.py 1942 runs/1942-assisted --decisions 4000 --assist
# Fresh-process replay with exact RAM checks every 120 game ticks:
& C:\msys64\opt\dreamerv3-python\python.exe scripts/stearcade_play.py dkong runs/dkong-replay --replay runs/dkong-play
```

Output directories must be new. The local checkout paths are declared at the top
of the script. The output manifest pins executable, TOS and private program
hashes, source hash, port revision and profile. The port checkout has existing
uncommitted changes; its revision alone is not the tested source identity.

## What the harness checks

An inserted private `dream_gate` stops once per DK NMI opportunity or 1942
interrupt delivery. Hatari dumps game RAM and blocks on the named pipe while
Dreamer chooses inputs or trains. One 12-tick macro is one decision. The first
128 decisions are seeded random warmup; later actions come from Dreamer. The
reward is changed-RAM fraction minus observed lives decreases. This is
exploration reward, not a demonstrated game-solving objective or a measured
learning advantage. Updates start at decision 64, every fourth decision.

Every tick verifies the private held SYSTEM/P1 bytes against the supplied
controls. Private code copies them into the port's input locations; this bypasses
host keyboard, joystick and pad handling. Live and replay use the same private
instrumentation. A watchdog detects missing game ticks/process exits; 600
identical whole-RAM observations trigger an anomaly. Neither check proves the
absence of subtler freezes. Raw RAM checkpoints are saved every 120 ticks.

The replay compares the full sampled RAM windows byte for byte. It checks
reproducibility of this port under this instrumentation; it is not an independent
arcade/reference equivalence test. Timelines include every explicit cheat write.

`--assist` refills lives. Donkey Kong also requests stage types 2, 3, 4 and 1 by
writing `$6227` every 4,800 ticks. A requested byte and a matching displayed stage
do not establish a natural clear. 1942 additionally uses invincibility `$E0A5`
and its existing finish-level cheat (`$EBD0/$E099/$E102`) every 1,200 ticks after
tick 1,600. Stage-byte changes are reported as such, not automatically as clears.
Cabinet coin/start attempts repeat every 6,000 ticks in both modes.

## Visual and scope limits

Headless Hatari debugger screenshots can be black even when the guest renders
valid game pictures. Use the guest planar framebuffer and palette instead.
The harness captures 256 lines at stride 256 and decodes STE planar pixels;
DK uses `display_screen_address`/`dk_palette`, 1942 uses
`flip_addr`/`shown_pal`. These are picture samples, not checks of border timing,
real display scanout, all sprite cases, or independent reference rendering.
Sound is disabled. Fast-forward host throughput is not STE performance evidence.

A complete-playability claim requires successful natural progression, the full
stage/ending or loop sequence, control modes, audio, and appropriate independent
logic/render checks. This bounded campaign explicitly leaves that claim false.

## 2026-10-20 campaign

Results are recorded in `stearcade-results-2026-10-20.md` after the retained
campaign and fresh-process replay complete. Small summaries can be retained;
generated binaries, checkpoints and compressed captures remain under ignored
`runs/`.
