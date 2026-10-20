# STEArcade playability check, 2026-10-20

**Neither game is certified completely playable without issues.** Donkey Kong's
bounded tests found no runtime failure, but the learned player did not clear the
opening stage naturally. 1942 hit a concrete untranslated-code failure after the
assisted stage counter advanced from 31 to 32.

## Campaign

Fresh per-game Dreamer agents trained and acted directly in Windows Hatari.
This used STE mode, 4 MB, CPU-exact/compatible execution, TOS 2.06, sound off,
headless video and fast-forward. The Hatari executable SHA256 is
`f113a3853fd993cac459cb672748f1c20c59ee4ac370501f49d5a6597cbf1796`.
Private instrumented builds preserve canonical ports and recordings. Full
method, controls, reward and scope: [playtest workflow](stearcade-playtesting.md).

| Game / run | Successful game ticks | Dreamer decisions / updates | Result |
|---|---:|---:|---|
| Donkey Kong, natural | 22,200 | 1,800 / 434 | Lives lost and cabinet restarts; stage type 1 only; no detected anomaly |
| Donkey Kong, assisted | 22,200 | 1,800 / 434 | Requested types 1→2→3→4→1; lives refilled; no detected anomaly |
| 1942, natural | 22,600 | 1,800 / 434 | Lives lost and cabinet restarts; stage byte 0 only; no detected anomaly |
| 1942, assisted | 41,649 | 3,387 / 831 | Stage bytes 0 through 32; fatal translated-code error, watchdog timeout |

The total discovery budget actually executed was 108,649 successful game ticks,
8,787 completed decisions and 2,133 learner updates. All four retained model
parameter sets are finite. This does not demonstrate learning improvement or
game-solving ability. Assisted visits and finish-cheat transitions are not
natural clears. The 1942 counter is reported as a byte, not as 33 distinct levels.

The two Donkey Kong recordings each replayed exactly at all 185 sampled RAM
checkpoints. Natural 1942 replayed exactly at all 188 checkpoints. Each replay
used a fresh Hatari process and the same private program hash. Every tick also
checked the supplied held input bytes. These are port reproducibility checks,
not original-arcade equivalence checks.

## 1942 failure

At tick 41,008, `$E103` changed from 31 to 32. The last successful observation
was tick 41,649. After the next input reply, the port wrote `program/CRASH.TXT`
and stopped reaching the game gate; the 45-second watchdog subsequently ended
the process. It was a fatal-code stop, not merely slow Dreamer training.

The crash starts with vector `0063` (decimal 99), the port's explicit
`z80_badpc`/`z80_badjump`/`z80_unsupported` fatal path. Saved D7 is `$0003C9C0`.
The swapped dispatch index decodes to Z80 target `$70F2`:

```text
index = 0x3c9c0 / 4 = 0xf270
target = swap16(0xf270) = 0x70f2
dispatch[index] = z80_badpc
```

The fixed ROM contains code at `$70F2` (`ld hl,$D450`, `ld de,$78C7`,
`call $6F48`, ...). The already translated `$7056` loads task address `$70F2`
into BC and `$7059` executes `rst $38`. The generated translation contains no
`Z70F2` entry, and the corresponding dispatch-table run is `z80_badpc`.
This establishes a missing translated task target in the tested build.
Natural reachability without the finish/lives/invincibility cheats has not
been tested. No translation or gameplay fix was applied during this check.

The fresh full assisted replay matched all 347 saved RAM checkpoints through
tick 41,640, then reproduced the same failure at tick 41,649 with a byte-identical
crash report and registers. There were 905 exact matching RAM checkpoints across
the four full replays. The three nonfailing replays passed; the fourth is a
reproduced failure, not a passing playability test.

The [crash text and registers](stearcade-1942-crash-2026-10-20.txt) and
[machine-readable results/manifests](stearcade-results-2026-10-20.json) are retained
with this report. Full input timelines remain in the local evidence folders.

## Reproduce

From the Dreamer repository, with the local checkout paths described in the
workflow guide:

```powershell
& C:\msys64\opt\dreamerv3-python\python.exe scripts/stearcade_play.py 1942 runs/1942-reproduce --replay runs/stearcade-1942-assisted
```

The script rebuilds the private program and requires its hash to match the
discovery manifest before replaying. The retained timeline includes all cheat
writes and the final input attempt. Do not overwrite the retained output folder.

## Evidence and remaining gaps

Local discovery folders are `runs/stearcade-{dkong,1942}-{natural,assisted}`;
replay folders append `-replay`. Each discovery retains its manifest, summary,
compressed tick inputs and RAM checkpoints, events, Dreamer model, build and
Hatari logs, and private source/map/program. This campaign additionally captured
the latest guest framebuffer/palette through the debugger; decoded previews
showed actual gameplay. Headless debugger screenshots were black and were not
used as evidence of blank guest rendering. The original screenshot probe and
subsequent raw-frame observer are diagnostic side effects only.

The tests bypass keyboard/pad input, disable sound, and do not establish full
renderer, scanout/border or original-logic equivalence. They do not exercise a
natural complete playthrough, all difficulty loops, two-player turns or all
control devices. Fast-forward wall time is not real STE performance evidence.
Donkey Kong therefore remains **partially tested**, and 1942 has a **failing
assisted late-game regression** that must be resolved before an issue-free
claim is supportable.

## Subsequent ending regression fixes

The STEArcade continuation fixed the missing `$70F2` task and then found and
fixed the companion high-score task entries `$6EEB` and `$6EDA`. The final
assisted build reached the ending/game-over path at tick 43,146, passed both
high-score task failures, and returned to stage byte 0 via explicit start/continue
inputs at tick 43,798. It progressed to byte 6 and stopped at tick 51,176.
A fresh process replayed the full retained timeline with 426 exact 4 KB RAM
checkpoints, identical transitions/markers, and no crash or input mismatch.
The ending displays “WE GIVE UP!”, a 10,000,000-point bonus, and “GAME OVER”;
the following start/continue is not an automatic difficulty loop.

Final private program SHA256:
`8b41d667769679be93edd708c6063f9986028afc2617245489e4407a794ef1df`.
Local evidence: `runs/stearcade-1942-ending-fixed-2` and
`runs/stearcade-1942-ending-verified`; runner:
`scripts/stearcade_1942_ending.py`. Detailed report and small results are in
`C:/Arbeit/STEArcade/docs/1942-ending-regression-2026-10-20.md` and `.json`.
The original failures above remain historical evidence. Natural complete
playability and independent arcade/render equivalence remain unverified.

## Subsequent Donkey Kong assisted loop verification

The retained assisted run was extended by 16,000 ticks using life refills,
exit-position assists and rivet removal. Unlike the original stage-selection
visits, this extension lets the ROM perform board-clear animations, advance its
board pointer and increment difficulty. It completed all four types in level 3,
entered level 4 at tick 28,859, and reached the repeating sequence's level 5 → 6
pointer wrap at tick 36,727. Gameplay was observed at level 6.

No missing translation or port failure was found; the private program hash
remained `891c50a0c0ef5f4f4d63f426b581d4c37d3c3cbb10f8f7e87579c0e1bee92b4f`.
A fresh Hatari process replayed all 38,200 retained inputs, matching 318 full
3 KB RAM checkpoints, identical transitions and all 22 framebuffer/palette
captures. No guest crash or input mismatch occurred. Full evidence is in
`runs/stearcade-dkong-loop-discovery-2` and `runs/stearcade-dkong-loop-verified`;
runner: `scripts/stearcade_dkong_loop.py`. Detailed report and small results:
`C:/Arbeit/STEArcade/docs/dkong-loop-regression-2026-10-20.md` and `.json`.

This remains assisted progression, with sound disabled and host controls
bypassed. Donkey Kong remains yellow, Verification pending, under STEArcade's
port-status legend; green natural complete playability is not established.
