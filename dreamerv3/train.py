"""Train and evaluate DreamerV3 without a display or Unix dependencies."""
import argparse
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from .agent import Agent
from .envs import make_env
from .replay import Replay


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(agent, task, episodes=10, seed=10000, random_policy=False):
    device = agent.cfg['device']
    rng = np.random.default_rng(seed)
    # Evaluation must not change training's latent sampling stream.
    devices = [torch.cuda.current_device()] if device.startswith('cuda') else []
    scores = []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        env = make_env(task)
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            state = None
            previous = torch.zeros(1, env.action_space.n, device=device)
            first, score, done = True, 0., False
            while not done:
                if random_policy:
                    action = int(rng.integers(env.action_space.n))
                else:
                    a, state = agent.act(torch.as_tensor(obs, device=device)[None], previous,
                                         torch.tensor([first], device=device), state, greedy=True)
                    previous, action = a, int(a.argmax(-1).item())
                obs, reward, terminated, truncated, _ = env.step(action)
                done, first = terminated or truncated, False
                score += reward
            scores.append(float(score))
        env.close()
    return dict(mean=float(np.mean(scores)), std=float(np.std(scores)), scores=scores,
                episodes=episodes, seed=seed)


def save_checkpoint(path, agent, replay, steps, cfg, best):
    payload = dict(agent=agent.state_dict(), optimizers=agent.optimizer_state(),
                   replay=replay.state_dict(), steps=steps, updates=agent.updates,
                   config=cfg, best=best, torch_rng=torch.get_rng_state(),
                   numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                   cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    os.replace(tmp, path)


def write_curve(path, rows):
    points = [(r['steps'], r['evaluation']['mean']) for r in rows if 'evaluation' in r]
    if not points:
        return
    low, high = min(y for _, y in points), max(y for _, y in points)
    span = max(high - low, 1)
    poly = ' '.join(f'{65 + 700*x/max(points[-1][0], 1):.1f},{325-260*(y-low)/span:.1f}' for x,y in points)
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="820" height="400" viewBox="0 0 820 400">'
                    f'<rect width="820" height="400" fill="white"/>'
                    f'<g font-family="sans-serif" font-size="15" fill="#253342">'
                    f'<text x="65" y="30">DreamerV3: held-out evaluation return</text>'
                    f'<path d="M65 55V325H780" fill="none" stroke="#8a95a0"/>'
                    f'<text x="10" y="70">{high:.1f}</text><text x="10" y="325">{low:.1f}</text>'
                    f'<text x="65" y="355">0</text><text x="700" y="355">{points[-1][0]:,}</text>'
                    f'<text x="330" y="385">Environment steps</text></g>'
                    f'<polyline points="{poly}" fill="none" stroke="#126dba" stroke-width="3"/></svg>',encoding='utf-8')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', default='pointmass', help='pointmass, cartpole, or importable module:factory returning a Gymnasium environment')
    p.add_argument('--steps', type=int, default=50000)
    p.add_argument('--logdir', type=Path, default=Path('runs/pointmass'))
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--envs', type=int, default=8)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--length', type=int, default=16)
    p.add_argument('--horizon', type=int, default=15)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--deter', type=int, default=128)
    p.add_argument('--latents', type=int, default=8)
    p.add_argument('--classes', type=int, default=8)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--gamma', type=float, default=0.997)
    p.add_argument('--lambda', type=float, default=0.95)
    p.add_argument('--entropy', type=float, default=3e-4)
    p.add_argument('--capacity', type=int, default=50000)
    p.add_argument('--warmup', type=int, default=1024)
    p.add_argument('--train-every', type=int, default=4, help='Collector ticks per gradient update')
    p.add_argument('--updates-per-train', type=int, default=1)
    p.add_argument('--eval-every', type=int, default=5000)
    p.add_argument('--eval-episodes', type=int, default=10)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--resume', type=Path)
    p.add_argument('--evaluate', type=Path, help='Evaluate checkpoint without training')
    p.add_argument('--require-learning', action='store_true', help='Exit nonzero if held-out verification fails')
    return p


def main():
    args = parser().parse_args()
    cfg = vars(args).copy()
    for key in ('logdir', 'resume', 'evaluate'):
        cfg[key] = str(cfg[key]) if cfg[key] is not None else None
    if min(args.batch, args.length - 1, args.envs, args.train_every, args.updates_per_train,
           args.eval_every, args.eval_episodes) <= 0:
        raise ValueError('Batch, envs, schedules must be positive; sequence length must exceed one')
    if args.capacity // args.envs < args.length:
        raise ValueError('Replay capacity must fit a full sequence per environment')
    torch.set_num_threads(args.threads)
    seed_all(args.seed)
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    checkpoint = None
    source = args.evaluate or args.resume
    if source:
        # Only load checkpoints you trust: they include optimizer and replay Python data.
        checkpoint = torch.load(source, map_location=args.device, weights_only=False)
        restore = ('task', 'hidden', 'deter', 'latents', 'classes') if args.evaluate else (
            'task', 'hidden', 'deter', 'latents', 'classes', 'envs', 'batch', 'length',
            'horizon', 'lr', 'gamma', 'lambda', 'entropy', 'capacity', 'warmup',
            'train_every', 'updates_per_train', 'seed')
        for key in restore:
            cfg[key] = checkpoint['config'][key]
            setattr(args, key, cfg[key])
    envs = [make_env(cfg['task']) for _ in range(args.envs)]
    obs_dim, actions = envs[0].observation_space.shape[0], envs[0].action_space.n
    agent = Agent(obs_dim, actions, cfg)
    if checkpoint:
        agent.load_state_dict(checkpoint['agent'])
    if args.evaluate:
        print(json.dumps(evaluate(agent, cfg['task'], args.eval_episodes, 20000), indent=2), flush=True)
        for env in envs:
            env.close()
        return
    logdir = args.logdir
    logdir.mkdir(parents=True, exist_ok=True)
    if not args.resume and (logdir/'metrics.jsonl').exists():
        raise FileExistsError('Run exists; use --resume or choose a fresh --logdir')
    (logdir/'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    replay = Replay(args.capacity, args.envs, obs_dim, actions, cfg['deter'],
                    cfg['latents'] * cfg['classes'], args.seed)
    steps, best = 0, -float('inf')
    if checkpoint:
        replay.load_state_dict(checkpoint['replay'])
        agent.load_optimizers(checkpoint['optimizers'])
        agent.updates, steps, best = checkpoint['updates'], checkpoint['steps'], checkpoint['best']
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        np.random.set_state(checkpoint['numpy_rng'])
        random.setstate(checkpoint['python_rng'])
        if checkpoint['cuda_rng'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([x.cpu() for x in checkpoint['cuda_rng']])
    hardware = dict(torch=torch.__version__, cuda=torch.version.cuda, device=args.device,
                    gpu=torch.cuda.get_device_name() if args.device.startswith('cuda') else None,
                    parameters=sum(p.numel() for p in agent.parameters()))
    (logdir/'hardware.json').write_text(json.dumps(hardware, indent=2), encoding='utf-8')
    print(json.dumps(hardware), flush=True)
    rows = []
    if args.resume:
        rows = [json.loads(line) for line in (logdir/'metrics.jsonl').read_text().splitlines()]

    def log(row):
        rows.append(row)
        with (logdir/'metrics.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)

    baseline = evaluate(agent, cfg['task'], args.eval_episodes)
    random_baseline = evaluate(agent, cfg['task'], 30, 20000, random_policy=True)
    log(dict(steps=steps, updates=agent.updates, evaluation=baseline, random_baseline=random_baseline))
    initial_verification = evaluate(agent, cfg['task'], 30, 20000)
    if args.resume and (logdir/'initial.json').exists():
        initial_verification = json.loads((logdir/'initial.json').read_text())
    else:
        (logdir/'initial.json').write_text(json.dumps(initial_verification, indent=2),encoding='utf-8')
    obs = np.stack([env.reset(seed=args.seed + i)[0] for i,env in enumerate(envs)])
    prev = np.zeros((args.envs, actions), np.float32)
    reward = np.zeros(args.envs, np.float32)
    first = np.ones(args.envs, np.float32)
    terminal = np.zeros(args.envs, np.float32)
    done = np.zeros(args.envs, bool)
    state = agent.world.initial(args.envs, args.device)
    episode_return = np.zeros(args.envs)
    recent = []
    start_time, last_log = time.monotonic(), time.monotonic()
    next_eval = (steps // args.eval_every + 1) * args.eval_every
    tick, metrics = 0, {}
    while steps < args.steps:
        replay.add(obs=obs, action=prev, reward=reward, first=first, terminal=terminal,
                   h=state[0].cpu().numpy(), z=state[1].cpu().numpy())
        with torch.no_grad():
            action, new_state = agent.act(torch.as_tensor(obs,device=args.device),
                                          torch.as_tensor(prev,device=args.device),
                                          torch.as_tensor(first,device=args.device), state)
        state = new_state
        chosen = action.argmax(-1).cpu().numpy()
        if steps < args.warmup:
            chosen = np.random.randint(actions,size=args.envs)
        prev = np.eye(actions,dtype=np.float32)[chosen]
        next_obs = []
        for i, env in enumerate(envs):
            if done[i]:
                value, _ = env.reset()
                reward[i], first[i], terminal[i], done[i] = 0, 1, 0, False
                prev[i] = 0
            else:
                value, r, term, trunc, _ = env.step(int(chosen[i]))
                reward[i], first[i], terminal[i] = r, 0, float(term)
                done[i] = term or trunc
                steps += 1
                episode_return[i] += r
                if done[i]:
                    recent.append(float(episode_return[i]))
                    recent = recent[-100:]
                    episode_return[i] = 0
            next_obs.append(value)
        obs = np.stack(next_obs)
        tick += 1
        if steps >= args.warmup and replay.count >= args.length and tick % args.train_every == 0:
            for _ in range(args.updates_per_train):
                batch, indices = replay.sample(args.batch, args.length, args.device)
                values, h, z = agent.update(batch)
                replay.refresh(indices, h, z)
                metrics = {k: float(v) for k,v in values.items()}
        if time.monotonic() - last_log >= 30:
            log(dict(steps=steps, updates=agent.updates, seconds=time.monotonic()-start_time,
                     train_return=float(np.mean(recent)) if recent else None, **metrics))
            last_log = time.monotonic()
        if steps >= next_eval or steps >= args.steps:
            result = evaluate(agent, cfg['task'], args.eval_episodes)
            log(dict(steps=steps, updates=agent.updates, seconds=time.monotonic()-start_time,
                     evaluation=result, **metrics))
            if result['mean'] > best:
                best = result['mean']
                torch.save(dict(agent=agent.state_dict(), config=cfg, steps=steps, updates=agent.updates),logdir/'best.pt')
            save_checkpoint(logdir/'latest.pt', agent, replay, steps, cfg, best)
            write_curve(logdir/'learning-curve.svg', rows)
            next_eval += args.eval_every
    # Verify selected policy on seeds not used to select checkpoints.
    best_checkpoint = torch.load(logdir/'best.pt',map_location=args.device,weights_only=False)
    agent.load_state_dict(best_checkpoint['agent'])
    final = evaluate(agent, cfg['task'], 30, 20000)
    delta = np.asarray(final['scores']) - np.asarray(random_baseline['scores'])
    lower = float(delta.mean() - 1.96 * delta.std(ddof=1) / np.sqrt(len(delta)))
    learned = final['mean'] > initial_verification['mean'] and lower > 0
    report = dict(task=cfg['task'], training_steps=steps, updates=agent.updates,
                  selected_checkpoint_steps=best_checkpoint['steps'], seed=args.seed,
                  initial_policy=initial_verification, random_policy=random_baseline,
                  trained_policy=final, paired_improvement_95pct_lower=lower,
                  learning_verified=bool(learned), elapsed_seconds=time.monotonic()-start_time,
                  criterion='Trained mean exceeds initial policy and paired normal-approximation 95% lower bound versus random exceeds zero. One training seed; not a paper reproduction.')
    (logdir/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)
    for env in envs:
        env.close()
    if args.require_learning and not learned:
        raise SystemExit('Learning verification failed; inspect verification.json')


if __name__ == '__main__':
    main()
