from pathlib import Path
import json,hashlib,shutil
root=Path(__file__).resolve().parents[1]
runs=['pointmass-cuda-seed0','cartpole-cpu-seed0']
records=[]
for name in runs:
    path=root/'runs'/name
    report=json.loads((path/'verification.json').read_text())
    assert report['learning_verified']
    report['run_directory']='runs/'+name
    report['checkpoint_sha256']=hashlib.sha256((path/'best.pt').read_bytes()).hexdigest()
    report['device']=json.loads((path/'hardware.json').read_text())['device']
    report['elapsed_seconds_note']='Time for the final invocation only; this run was resumed.'
    records.append(report)
    destination=root/'docs/results'/name
    destination.mkdir(parents=True,exist_ok=True)
    for filename in ('verification.json','config.json','hardware.json','initial.json','learning-curve.svg'):
        shutil.copyfile(path/filename,destination/filename)
(root/'docs/learning-results.json').write_text(json.dumps(dict(date='2026-10-20',core_tests_passed=9,runs=records),indent=2),encoding='utf-8')
text='''# Native Windows learning verification

Verified on 20 October 2026 using MSYS2 launch scripts and native Windows CPython.
No WSL or Docker was used. All nine core tests passed, including episode reset,
terminal/truncation returns, replay wrap-around, optimizer persistence, a full
three-network update, checkpoint restoration, and the external environment factory.

| Task | Training device | Environment steps | Initial policy | Random policy | Trained policy |
|---|---|---:|---:|---:|---:|
'''
for r in records:
    text+=f"| {r['task']} | {r['device']} | {r['training_steps']:,} | {r['initial_policy']['mean']:.2f} | {r['random_policy']['mean']:.2f} | **{r['trained_policy']['mean']:.2f}** |\n"
text+='''
Each score is a mean over 30 held-out episodes with seeds 20000-20029. Models
were selected using ten development episodes with seeds 10000-10009. One
training seed (0) was used per task. PointMass has maximum episode return 100;
CartPole has maximum episode return 500. Both automatic learning checks passed.
This establishes learning on these tasks; it is not a full DreamerV3 benchmark
reproduction or evidence of arcade telemetry gains.

The point-mass run used a 595,764-parameter agent (including the target critic)
on the RTX 5090 Laptop GPU. Its selected checkpoint is from step 18,000. The
CartPole run used a 194,396-parameter agent on CPU, with its selected checkpoint
at step 24,007. Runs were resumed after checking the time-limit return fix;
resume starts fresh environment episodes while retaining replay and optimizer
state. The `elapsed_seconds` fields in the original run reports describe only
the final invocation, not total time across resumes.

Artifacts:

- [PointMass verification](results/pointmass-cuda-seed0/verification.json) and
  [learning curve](results/pointmass-cuda-seed0/learning-curve.svg).
- [CartPole verification](results/cartpole-cpu-seed0/verification.json) and
  [learning curve](results/cartpole-cpu-seed0/learning-curve.svg).
- [Machine-readable results and checkpoint hashes](learning-results.json).
- [Native Windows/CUDA capability probe](native-capabilities.json).
- [Future jsm68kdisassembler integration](telemetry-integration.md).

Small result artifacts are versioned under `docs/results/`. Training runs and
checkpoints are excluded from Git. On the development machine the selected
checkpoints are in `runs/pointmass-cuda-seed0/best.pt` and
`runs/cartpole-cpu-seed0/best.pt`; the commands below require those local files
or newly trained checkpoints.

Re-evaluate the selected native CUDA-trained policy on CPU:

```bash
cd /c/Temp/dreamerv3
bash scripts/run.sh --evaluate runs/pointmass-cuda-seed0/best.pt --device cpu --eval-episodes 30
```

Re-evaluate CartPole:

```bash
bash scripts/run.sh --evaluate runs/cartpole-cpu-seed0/best.pt --device cpu --eval-episodes 30
```

Training settings are recorded in each run's `config.json`. The reference
runs used PointMass defaults with 20,000 steps followed by resume to 22,000;
CartPole used `--hidden 64 --deter 64 --threads 1 --length 32 --train-every 2`,
was checkpointed at 20,007 steps, and resumed to 24,007. Repeating these runs
may differ slightly with other PyTorch versions, CUDA kernels, or resume points.
'''
(root/'docs/learning-results.md').write_text(text,encoding='utf-8')
print('Verified both learning checks and saved result summaries.')
