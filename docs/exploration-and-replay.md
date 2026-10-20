# Exploration and replay roadmap

Build a reusable exploration-and-replay system around Dreamer before extending
the learner. The existing RAM vectors and discrete controller macros fit the
current network interface. This is a plan, not a report of completed emulator
integration or measured arcade results.

## Implementation update (2026-10-20)

F030Arcade-private now contains native Windows MAME/Hatari adapters, portable
timelines, archive-prefix returns, Dreamer exploration and replay validation.
The validated telemetry overlay has been promoted into ffight's baseline;
disassembly, immediate-promotion checks and the 4 MB byte-exact rebuild pass.
The coverage study found a small replay-confirmed Dreamer contribution beyond
random exploration, rather than a general learned-policy advantage.

The retained 5,040-source-tick opening-stage branch passes 70 exact game-state
checks in Hatari on two independent replays, with 2,010 dense boundary checks
and 13 adapter tests also passing. Replay aligns actual guest input reads;
VBL counts are separate timing evidence. Original gameplay-state expectations
and zero tolerances are preserved. This is not a full-game completion.
The implementation and authoritative findings are in
[F030Arcade-private's Final Fight progress](https://github.com/AnimaInCorpore/F030Arcade-private/blob/stearcade-falcon-ports/games/ffight/PROGRESS.md).
The priorities below remain the longer-term roadmap.

## First milestone: Final Fight

**Dreamer-guided archive exploration produces additional useful telemetry, and
one discovered timeline replays in Hatari with checked state checkpoints.**

Acceptance is based on observed coverage and reproducible port behavior.
Dreamer's predicted reward, training return, and model loss are diagnostics,
not acceptance criteria.

### Acceptance gates

1. **Reliable native execution.** Pin the Windows MAME and Hatari builds, game
   and port revisions, ROM/image hashes, configuration, cheats, action timing,
   and initial states. Exercise stepping, input injection and release,
   observation reads, deadlines, EOF/error handling, process cleanup, and
   artifact collection. Verify the selected Hatari build's automation controls
   before depending on them.
2. **Measured additional telemetry.** Freeze a baseline capture and archive.
   Compare random exploration, existing Go-Explore exploration, and
   Dreamer-guided exploration under equal emulator-frame budgets and the same
   return-state schedule. Keep training, selection, and final evaluation
   coverage sets separate. Report new instruction starts, CODE,
   READ_BYTE/WORD/LONG, and ADDRESS evidence separately, plus archive depth and
   resulting disassembly changes. Require a nonempty, replay-confirmed useful
   delta against the frozen baseline; report incremental gain against the
   archive-only comparator without assuming Dreamer wins. Fix budgets, seeds,
   and the useful-evidence review rules before final evaluation.
3. **Portable discovered timeline.** Retain at least one timeline discovered
   during Dreamer-guided exploration, with logical controls and game-tick
   timing, explicit releases, initial-state identity, restore boundaries,
   cheat metadata, and any bootstrap inputs needed to reach its starting state
   in each emulator. A MAME save state is not a portable Hatari starting state.
   Replay in a fresh MAME process to confirm the claimed telemetry delta.
4. **Checked Hatari replay.** Replay that scenario in the Falcon port and
   produce machine-readable expected/observed results at multiple declared
   state checkpoints. Align by game-owned stage, scene, input offset, and
   object counts, with bounded alignment deadlines; host VBL alone is
   insufficient. Compare selected RAM fields using explicit field mappings,
   pointer normalization, and justified tolerances. Missing or unreachable
   checkpoints fail the check rather than silently passing.
5. **Reproducible failure output.** If replay diverges, preserve the earliest
   checked mismatch or alignment timeout, expected and observed state,
   registers, relevant memory, input history, and an exact rerun command.
   Reproduce it in a fresh process and minimize the input sequence while
   preserving the same failure signature. Keep both the original discovery
   and minimized regression case. A fully matching replay is a valid result;
   finding a port bug is not mandatory, and a mismatch is not a passing
   correctness check merely because it is reproducible.

The milestone evidence bundle contains the run manifest, baseline and output
hashes, coverage delta and disassembly review, archive provenance, portable
timeline, checkpoint specification and comparison report, logs, and rerun
commands. Include any failure bundle and minimized regression. Report frame
budget, wall time, training time, and restore count separately. Run the arcade
project's preview and immediate-promotion checks before merging telemetry into
its baseline.

## Implementation priorities

| Priority | Work | Deliverable |
|---|---|---|
| 1 | Native Windows MAME/Hatari adapters | Reliable stepping, controls, observations, deadlines, cleanup, artifacts |
| 2 | Portable input timelines | Versioned logical-input records replayable in both emulators |
| 3 | Go-Explore archive integration | Repeatable frontier restoration and exploration with timeline provenance |
| 4 | Coverage-aware observations and rewards | Query age, archive-cell context, bounded coverage summaries, separate novelty measurements |
| 5 | Differential test harness | State alignment, RAM comparison, first divergence capture, failure minimization |
| 6 | Sequence replay and model validation | Better reuse and validation of expensive emulator experience |

Use the arcade project's `docs/go-explore.md` as the archive workflow authority
and `DEBUGGING.md` for state alignment and renderer checks. On each archive
restore, reset Dreamer's recurrent state or restore the matching context.
Record actual observations, actions, restore boundaries, continuation flags,
and frame counts. Coverage novelty depends on a changing visited set and may
arrive late; expose that context rather than treating reward as stationary.

For Hatari, startup debugger scripts via `--parse` are a candidate automation
mechanism to verify against the selected Windows binary; they do not by
themselves establish deterministic stepping or input replay. Consult the
[Hatari debugger manual](https://hatari.tuxfamily.org/doc/debugger.html) during
adapter implementation.

## Learner improvements after the integration foundation

- Complete online trajectory mixing in replay.
- Support longer sequences and configurable model sizes for delayed effects.
- Persist or reconstruct emulator/archive state, coverage history, episode
  schedule, recording state, and recurrent context in experiment checkpoints.
- Validate multi-step predictions against held-out real emulator trajectories.
- Make replay ratio configurable to reuse expensive collected experience.

Defer image support until RAM-based exploration demonstrates measurable value.
Continuous actions are not a priority for the existing button interface.

See [the integration notes](telemetry-integration.md) for the inspected adapter
interface, native Windows transport gap, and evaluation isolation requirements.
