# cc-baseline

A local flight recorder for Claude Code. It reads the transcripts Claude Code already writes to `~/.claude/projects`, measures how Claude behaves in each session, and tells you when that behaviour shifts after an update, a model switch or a change to your CLAUDE.md, skills or settings.

"It got worse after the update" is hard to act on with no numbers behind it. cc-baseline gives you the numbers from your own sessions, with intervals that tell you whether a change is real or noise.

- **Zero tokens, zero network.** Pure Python standard library. Nothing leaves your machine.
- **Honest statistics.** Session-level bootstrap errors, corrected for the number of tests so the chance of any false alarm stays near 5%. A change is reported only when its interval excludes zero and a permutation test agrees, and everything else that changed at the same time is listed next to it.
- **Keeps history.** Claude Code deletes old transcripts after a while. cc-baseline keeps the counts in a small local SQLite file, so your baseline survives.

## Install

Needs Python 3.10 or newer.

As a Claude Code plugin (adds the session-start alert):

```
/plugin marketplace add JohnNasr/cc-baseline
/plugin install cc-baseline@cc-baseline
```

As a command-line tool (for reports):

```
pip install git+https://github.com/JohnNasr/cc-baseline
```

## Use

```
cc-baseline ingest                 # read new or changed transcripts (also runs before every report)
cc-baseline report                 # metrics for your newest Claude Code versions; (n=...) marks fewer than 20 events
cc-baseline report --by model      # ... or by model, effort, entrypoint, permission, rules, skills
cc-baseline drift                  # find version shifts and test your newest model/effort/rules/skills
cc-baseline drift --md             # the same as markdown you can paste into a GitHub issue
```

Common options: `--all` includes subagent and SDK/headless sessions (left out by default), `--project TEXT` keeps only projects whose path contains TEXT, `report --window DAYS` keeps only the last N days of recorded sessions (counted back from the newest one), `drift --days N` sets the window on each side of a change (default 14).

Example `drift` output (illustrative numbers):

```
Claude Code version: tested 5 version changes with 14-day windows on each side (2 skipped for too little data), intervals corrected for 50 tests (part of a 5% chance of any false alarm in the whole run).
  at 2.2.0 (from 2026-01-11; 35 vs 53 sessions):
    worse   Edits without a prior Read: 0.8% -> 13.4% (+12.6 pts, interval +3.6 pts .. +23.1 pts)
    better  Tool calls that errored: 9.3% -> 3.9% (-5.4 pts, interval -9.3 pts .. -2.2 pts)
    Also different between the two groups: model (model-a 77% -> model-b 68%); permission mode (default 71% -> auto 98%)
```

With the plugin installed, a new session starts with one line when a version shift makes a metric worse, once per shift:

```
cc-baseline: possible behaviour shift: edits without a prior read 0.8% -> 13.4% since 2.2.0. Details and caveats: `cc-baseline drift` (CLI: pip install git+https://github.com/JohnNasr/cc-baseline).
```

The hook tests only version changes from the last 28 days whose "after" window has ended, with 7-day and 14-day windows (half the 5% error budget each). Each check repeats the same test instead of looking again at growing data, so an alert comes 7 to 14 days after a version takes over (`cc-baseline drift` looks sooner), and an old shift you already know about stays quiet. It parses for at most 15 seconds per session start and stops testing at 25 seconds, newest changes first. It leaves transcripts over 100 MB to `cc-baseline ingest`, and says nothing until the store is complete (a large first import can take a few sessions). It never blocks a session: any error is swallowed. On Windows the hook needs Git Bash (part of Git for Windows), which Claude Code uses for hook commands; without it the hook stays silent. The `drift` command in the alert needs the command-line install.

## What it measures

Every metric is a ratio of totals over a group of sessions (for example all edits, not an average of per-session rates), so long sessions weigh more than short ones.

| Metric | Counts | Worse when |
|---|---|---|
| Reads per edit | Read calls / edit calls | lower |
| Edits without a prior Read | Edit/MultiEdit/NotebookEdit calls (and Writes over existing files) on a file not read earlier in the session / edit calls | higher |
| Shell used for file reads/writes | Bash/PowerShell calls where any command (split at `&&`, `\|\|`, `;` or a new line) starts with `cat`, `head`, `sed`, `grep`, `find`, `Get-Content`, `echo > file` and similar / shell calls | higher |
| Tool calls that errored | tool results marked as errors / tool calls | higher |
| 'Done' with no check after last edit | turns that changed files and ended with "done", "fixed", "all tests pass" etc. but ran no test, lint, build or type check after the last edit / such turns | higher |
| Prompts you interrupted | interrupts / prompts | higher |
| Tool calls per prompt | tool calls / prompts | (shown, no direction) |
| Subagents per prompt | Agent/Task calls / prompts | (shown, no direction) |
| Output tokens per prompt | output tokens / prompts | (shown, no direction) |
| Cache-write tokens per prompt | cache-creation tokens / prompts | (shown, no direction) |

Each session is also fingerprinted by Claude Code version, model, effort, entrypoint, permission mode, a hash of the CLAUDE.md and rules files Claude was given, and a hash of the skill listing. A session resumed after an update is split into one segment per version.

## How the statistics work

- **Intervals.** `report` resamples sessions with replacement (bootstrap, 1000 draws) and shows half the width of the 95% interval, with no interval for groups under 5 sessions. `drift` compares two groups with a t interval (Welch degrees of freedom) on the difference; its standard error comes from 500 bootstrap draws per group but never drops below the binomial (percentages) or Poisson (other metrics) error at the pooled rate, so a group with zero events cannot look certain. Far in the tail, where the correction below puts the cut-off, skewed per-session counts make any such interval too narrow, so a change must also pass a permutation test (shuffling sessions between the two groups) at the same level.
- **Multiple tests.** Every metric at every tested point in your history is one test. Intervals are Bonferroni-corrected for all of them, and a `drift` run splits its 5% between its five sections (version, model, effort, rules, skills), so the chance that a run shows any false alarm is at most about 5%.
- **Minimums.** A metric is compared only when both groups have at least 20 events in its denominator, and a group needs at least 5 sessions (8 for version scans).
- **Version shifts.** Claude Code ships often, so the newest version alone rarely has enough sessions. `drift` tests every point where a version newer than any before it took over: the first session from which most of the next 8 sessions run it or newer. A single trial session days earlier is not the change, and opening an old install later is not one either. It compares the 14 days before it with the 14 days from it. Later versions count in the "after" window, since a change usually persists. For each metric and direction it keeps the clearest point, and it calls results "possible shifts".
- **Switches.** For model, effort, rules and skills, `drift` compares sessions on the newest value against sessions on other values in the 14 days before it first appeared. Sessions with no recorded value for that field are left out.
- **Confounders, not corrections.** If the two groups also differ in model, effort, entrypoint, permission mode, rules, skills or project mix, that is listed next to the result. cc-baseline does not try to adjust for it: read a version shift that coincides with a model switch as "one of these changed things".

## Privacy

cc-baseline reads only local transcript files and makes no network calls. The local store (`~/.cc-baseline/baseline.db`) holds per-session counts, the fingerprint fields above, the session id, timestamps and the first working directory. It never stores prompt text, code, file contents or tool output; rules and skills are stored as 10-character hashes.

`drift --md` leaves out project names and paths, so its output is safe to paste into a public issue. Model names, Claude Code versions, permission modes and the rules/skills hashes do appear. Read it before you post it.

## Limitations

- The metrics are heuristics on transcript records. "Done" claims and shell file operations are matched by word lists and will miss or misread some cases.
- It favours quiet over sensitive. In simulations with 4 sessions a day and a new version every 3 days, running the hook every third day for 120 days raised a false alarm in 1 of 24 unchanged histories. It caught a six-fold rise in edits without a prior Read (6% to 37%) in 24 of 24 histories, but a tripled one (6% to 19%) in none. A full `drift` scan over 50-day histories resampled from real sessions (about 155 tests each) raised a false alarm in 5 of 300. Small shifts need more sessions than most people have in two weeks.
- Results are correlations in your own usage. Your projects and habits change over time too; check the confounder line before blaming a release.
- The transcript format is undocumented and changes between Claude Code versions. Records cc-baseline does not recognise are skipped, not guessed.
- Sessions with no prompts (for example empty or purely automated transcripts) are ignored.
- When a new cc-baseline counts a metric differently, stored transcripts are re-read. Those whose files Claude Code has already deleted cannot be, so they are left out; `cc-baseline ingest` says how many.

## Settings

| Variable | Effect |
|---|---|
| `CC_BASELINE_DB` | where the SQLite store lives (default `~/.cc-baseline/baseline.db`) |
| `CLAUDE_CONFIG_DIR` | where Claude Code keeps its data; transcripts are read from its `projects` folder (default `~/.claude/projects`) |

`--db` and `--root` override both per command.

## Development

```
pip install -e .[dev]
pytest -q
```

MIT license.
