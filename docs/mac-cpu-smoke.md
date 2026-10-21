# Mac CPU smoke test — 20 October 2026

Verified on an Apple M1 MacBook Air, 8 GB unified memory, macOS 27.0.1,
with native arm64 Python 3.11.17 in `.venv`, PyTorch 2.8.0, NumPy 2.2.6,
and Gymnasium 1.2.1. `pip check` passed. MPS reports available, but was not tested.

`bash scripts/test.sh` passed all nine core tests in 1.030 seconds.
The short CPU command in the README completed 2,048 actual CartPole steps and
119 optimizer updates with finite reported losses in 5.86 seconds (the trainer's
timer excludes initial baseline evaluations and process startup).
It produced configuration/hardware records, metrics, a learning curve,
`best.pt`, `latest.pt`, and the held-out verification report under
`runs/mac-cpu-smoke/`. Generated runs and the virtual environment remain ignored.

Mean held-out returns over 30 episodes: initial 13.17, random 22.53, trained 9.40.
`learning_verified` is false. This is an execution smoke test, not evidence of
learned control; the short command intentionally omits `--require-learning`.
It does not validate arcade exploration, emulator replay, or MPS execution.

Re-evaluate the retained checkpoint:

```bash
bash scripts/run.sh --evaluate runs/mac-cpu-smoke/best.pt --device cpu \
  --threads 1 --eval-episodes 3
```
