# DreamerV3 on native Windows

A compact PyTorch implementation of the DreamerV3 learning algorithm, developed
through `C:\msys64` and executed by native Windows CPython. No WSL or Docker is
required. Start with a damped point mass reaching a randomly chosen goal, then
use the included standard Gymnasium CartPole benchmark.

The intended later application is telemetry exploration for
`C:\Arbeit\F030Arcade-private\third_party\jsm68kdisassembler`. See
[the integration notes](docs/telemetry-integration.md) for the inspected RAM/action
interface, native Windows transport work, archive exploration, and coverage
validation requirements. The native adapters and exploration runner now live
in that arcade repository.

The first arcade milestone is [Final Fight exploration and checked Hatari replay](docs/exploration-and-replay.md),
judged by useful coverage gained and reproducible port behavior rather than
Dreamer's predicted reward.

The retained opening-stage branch now passes 70 exact game-state checks in
Hatari on two independent replays. This does not demonstrate a full-game
playthrough; see the roadmap's implementation update for scope and evidence.

## Verified learning

On this machine, the CUDA-trained point-mass agent reached **94.25/100**, versus
**-81.64** for random actions. The CPU-trained CartPole agent reached
**453.53/500**, versus **22.53** for random actions. These are means over 30
held-out episodes, with one training seed per task. All nine core tests passed.
See [the results and checkpoint details](docs/learning-results.md). Small result
summaries are included in the repository; generated runs and checkpoints are
excluded from Git.

## Installation

From an MSYS2 shell:

```bash
cd /c/Temp/dreamerv3
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1
bash scripts/test.sh
```

`setup.ps1` installs an isolated CPython 3.12.10 runtime at
`C:\msys64\opt\dreamerv3-python` and PyTorch 2.8.0 with CUDA 12.8 wheels,
NumPy 2.2.6, and Gymnasium 1.2.1. It does not change your system Python or PATH.
MSYS2's MinGW Python uses a different extension ABI from official Windows
PyTorch wheels; the separate native CPython runtime avoids that incompatibility.
The embedded interpreter intentionally has an isolated module search path; use
the supplied launch scripts, which add the project directory explicitly.

Check the native runtime and execute a real CUDA matrix operation:

```bash
/c/msys64/opt/dreamerv3-python/python.exe scripts/capabilities.py
```

Tested hardware: NVIDIA GeForce RTX 5090 Laptop GPU, compute capability 12.0,
24 GB VRAM, Windows 11, driver 617.42. The installed CUDA toolkit is 13.4;
PyTorch uses its bundled CUDA 12.8 runtime and has native `sm_120` support.

## Train

Small physical control task (maximum episode return 100):

```bash
bash scripts/run.sh --task pointmass --device cuda --steps 50000 \
  --logdir runs/pointmass --require-learning
```

The point mass observes position, velocity, and goal. It applies left, neutral,
or right thrust to damped dynamics. Each episode lasts 100 steps. The reward is
`1 - (position - goal)^2 - 0.1 * velocity^2`. Targets and initial conditions vary
by episode. There are no demonstrations, controller labels, policy warm starts,
or task-specific model equations inside the agent.

Standard CartPole (unaltered environment reward and dynamics):

```bash
bash scripts/run.sh --task cartpole --device cpu --hidden 64 --deter 64 \
  --threads 1 --length 32 --train-every 2 --steps 60000 \
  --logdir runs/cartpole --require-learning
```

Small networks can train faster on CPU because individual CUDA operations have
launch overhead. Both devices execute the same model and learning algorithm.
Use `--device cuda` for GPU training. `--steps` counts actual environment steps,
excluding internal reset records; it may be exceeded by up to one collector tick.

Each run produces:

- `config.json` and `hardware.json`: settings and runtime information.
- `metrics.jsonl`: model losses, training returns, and independent evaluations.
- `learning-curve.svg`: evaluation return versus collected experience.
- `best.pt`: the policy/model selected using development evaluation seeds.
- `latest.pt`: model, optimizers, replay, random states, and training progress.
- `initial.json` and `verification.json`: initial and final held-out results.

Final verification evaluates 30 episodes with seeds 20000-20029, separate from
the seeds used for checkpoint selection. It compares against the initial agent
and a random policy. `--require-learning` exits unsuccessfully if the final mean
does not exceed the initial mean, or if the paired normal-approximation 95%
lower bound for improvement over random is not positive. This is a basic
learning check for one training seed, not a claim of statistical replication
across training seeds or of solving a benchmark.

Evaluate a saved policy on CPU or GPU:

```bash
bash scripts/run.sh --evaluate runs/pointmass/best.pt --device cpu --eval-episodes 30
```

Resume training to a larger total step count:

```bash
bash scripts/run.sh --resume runs/pointmass/latest.pt \
  --logdir runs/pointmass --steps 100000 --device cuda
```

Resume restores the original architecture, task, optimizer/replay settings,
random states, and progress. Environment instances restart at fresh episode
boundaries, so resumed execution is not bit-for-bit identical to uninterrupted
training. Use the same run directory. Load only trusted checkpoints, as PyTorch
checkpoint files can include executable Python objects.

## Implemented algorithm

- Vector encoder/decoder using symlog observations and normalized SiLU MLPs.
- RSSM with eight recurrent blocks and straight-through categorical latents.
- KL dynamics/representation balancing (1.0/0.1), one free nat, 1% uniform mix.
- Observation reconstruction, twohot reward prediction, continuation prediction.
- Latent imagination, lambda returns, REINFORCE actor, entropy regularization.
- Distributional critic with EMA regularization and replay value training.
- Exponential twohot bins, zero-initialized reward/value output layers.
- Return normalization using smoothed 5th/95th percentiles and a minimum range.
- Adaptive gradient clipping and bias-corrected LaProp.
- Sequence replay with recent non-overlapping sequence inclusion, stored latent
  initialization, and refreshed replay latent states.
- Reset masking, terminal bootstrapping, and time-limit truncation handling.

The collector stores terminal observations before inserting reset observations.
True terminations teach zero continuation; time limits retain continuation.
Replay entries associate each observation with its incoming action and reward.
The world model trains on real experience; actor updates use imagined experience.

The default small configuration has hidden/deterministic widths 128/128 and
8 categorical latents with 8 classes each. It uses learning rate `3e-4`, batch
shape `16 x 16`, imagination horizon 15, replay capacity 50,000, and one update
every four collector ticks with eight environments. These are deliberately
smaller/faster development settings than the paper's model and training budget;
the paper reports learning rate `4e-5`, batch `16 x 64`, and replay capacity 5M.
The online replay implementation includes one fresh sequence per batch when
available; it does not implement the paper's complete multi-trajectory online
queue. The recurrent initialization is zero rather than learned.

This first implementation supports vector observations and discrete actions.
Image encoders/decoders, continuous action distributions, Minecraft, distributed
workers, and the full benchmark harness are not implemented. It is an independent
implementation guided by [the local paper](docs/2301.04104v2.md), not a copy of
the author's JAX implementation, and it does not claim paper-level reproduction.

## Source layout

| File | Purpose |
|---|---|
| `dreamerv3/networks.py` | Encoder, block GRU, RSSM, categorical distributions |
| `dreamerv3/maths.py` | Transforms, twohot loss, returns, AGC/LaProp |
| `dreamerv3/agent.py` | World model, imagination, actor and critic updates |
| `dreamerv3/replay.py` | Sequence ring buffer and latent refresh |
| `dreamerv3/envs.py` | Point mass and CartPole factories |
| `dreamerv3/train.py` | Collection, evaluation, checkpointing, learning check |
| `tests/test_core.py` | Numerical, reset, replay, optimization and persistence tests |

References: [DreamerV3 paper](https://arxiv.org/abs/2301.04104v2),
[author's implementation](https://github.com/danijar/dreamerv3),
[native Windows PyTorch installation](https://pytorch.org/get-started/locally/).
