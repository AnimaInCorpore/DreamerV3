# Future F030Arcade telemetry integration

Target: `C:\Arbeit\F030Arcade-private`, particularly
`third_party/jsm68kdisassembler`. This document records an integration direction;
the initial inspection itself modified no arcade files and launched no emulator.
Implementation has since proceeded in F030Arcade-private; see the roadmap's
implementation update for measured coverage and replay results.

The [exploration and replay roadmap](exploration-and-replay.md) defines the
first Final Fight milestone, implementation priorities, and acceptance gates
for useful telemetry and checked Hatari replay.

## Existing interface inspected on 2026-10-20

- `scripts/ppo-telemetry/ppo_telemetry.py` already implements a Gymnasium
  environment over telemetry-instrumented MAME and `bridge.lua`.
- Its observation is two consecutive 1,024-byte RAM samples, scaled to [0,1]
  and concatenated into a 2,048-element vector.
- Its action is a discrete controller macro: directions and pulsed buttons,
  usually over 12 frames. Profiles support a third button with a wider input
  wire format.
- The present reward is `0.1 * new_instruction_starts + 0.002 * reported_PCs -
  0.01`. Breakpoint queries normally occur every fourth decision, so some rewards
  arrive up to three decisions late. The reported set is not a dense trace of
  every PC execution: it comes from one-shot breakpoint hits on candidate ROM
  instruction starts.
- Each episode uses a private MAME working directory. On clean exit, generated
  sidecars are OR-merged into an output capture, leaving committed baseline
  sidecars separate.
- `docs/go-explore.md` describes the recommended archive/return/explore loop.
  It already supports returning to saved game states and accumulating telemetry
  across branches. The authoritative workflow and port-specific results remain
  in that repository.

These vector observations and categorical actions fit the current Dreamer
network interface. A visual world model is not a prerequisite for this route.

## Integration steps

1. Build a small environment adapter that owns MAME processes, exposes the
   existing RAM features/actions, and closes processes reliably on every exit.
   Keep emulator transport separate from policy code. The trainer accepts
   `--task package.module:factory`, where the no-argument factory returns a
   Gymnasium environment with a one-dimensional Box and Discrete actions.
   The factory module must be importable from the project or installed runtime.
   It may read a separate JSON configuration for the game/profile/output path.
2. Port the transport to native Windows. The current bridge uses
   `select.select()` on a subprocess stdout pipe, which Windows' socket select
   cannot service. Use a reader thread plus a bounded queue, or Windows overlapped
   pipe I/O, with deadlines, EOF handling and stderr logging. Use the verified
   native telemetry MAME executable, not an assumed Unix path. Audit ROM staging
   links for Windows support as well.
3. Make exploration context observable. Coverage novelty is relative to the
   baseline and a growing visited set. Identical RAM can therefore yield different
   novelty rewards at different times. Add bounded coverage/visit summaries,
   archive-cell context, and query-age information to observations, or use a
   clearly specified stationary reward proxy. Record the choice in checkpoints.
   Avoid feeding a huge raw ROM coverage bitmap to the MLP without measurement.
4. Let the Go-Explore archive choose return states, then let Dreamer propose
   exploration actions from those states. On state restoration, reset the RSSM
   or restore the matching recurrent context; do not carry latent memory from an
   unrelated archived branch. Store actual observations, actions, rewards,
   continuation flags, restore boundaries, and frame counts in replay.
5. Train a fresh per-game world model and policy. PointMass/CartPole checkpoints
   establish that the learner functions; their weights are not arcade models.
   Start with one environment because emulator memory and startup costs can
   dominate. Tune model/batch size only after measuring native throughput.
6. Compare random actions, the existing Go-Explore method, and Dreamer-guided
   archive exploration under equal emulator-frame budgets and separately report
   wall time, model-training time, and number of state restores. Keep the emulator
   binary/core revision, baseline sidecar hash, game, seed, cheats and action
   timing identical across comparisons. A new core may mark more telemetry even
   without improved exploration.
7. Evaluate useful telemetry: new CODE, READ_BYTE/WORD/LONG and ADDRESS evidence,
   deeper/different archive cells, and changes to disassembly structure. Preserve
   complete input timelines and all state-restoration/cheat metadata. Confirm
   candidate captures by deterministic replay in a fresh process, then run the
   project's preview and immediate-promotion checks before any baseline merge.

## Evaluation boundaries

Coverage evaluation consumes novelty. Use separate processes, copied baseline
coverage sets, archives and output captures for training, checkpoint selection,
and final evaluation. Do not share an environment's `global_seen` or accumulator
between those roles. Keep initial and trained comparisons at the same baseline.
For archive-assisted runs, report coverage attributable to the whole method and
the incremental gain beyond the same return-state/archive selection alone.

The existing adapter varies starting floors/checkpoints by episode number and
does not use `reset(seed=...)` to choose them. A fair comparison must explicitly
pin and restore that schedule as part of the experiment state; setting Python
random seeds alone is insufficient.

The standalone trainer restarts environments on resume, which is fine for the
control demonstrations. A telemetry adapter will also need to persist or
reconstruct its coverage set, archive, episode schedule and recording state.
The archive and accepted telemetry evidence remain the durable outputs; a higher
learned return alone is not proof of better disassembler coverage.
