"""cc-baseline command line: ingest, report, drift, hook."""
import argparse
import json
import os
import sys
import time
from datetime import timedelta

from . import __version__
from .stats import ALPHA, DIMS, METRICS, MIN_EVENTS, drift, group_cis, parse_ts, scan, segments, select, totals, value
from .store import default_db, default_root, get_meta, ingest, load_sessions, open_db, set_meta, stale_files

HOOK_INGEST_S = 15  # the plugin gives the hook 30 s in all
HOOK_TOTAL_S = 25
HOOK_WINDOWS = (7, 14)  # days on each side; each gets half the error budget
HOOK_LOOKBACK = 28  # older version changes had their chance
INSTALL = "pip install git+https://github.com/JohnNasr/cc-baseline"
SWITCH_DIMS = ["model", "effort", "rules", "skills"]
TAGS = {True: "worse", False: "better", None: "changed"}


def fmt(metric, v):
    if v is None:
        return "-"
    if metric[4] == "pct":
        return f"{v * 100:.1f}%"
    if metric[4] == "int":
        return f"{v:,.0f}"
    return f"{v:.2f}"


def fmt_delta(metric, d):
    if metric[4] == "pct":
        return f"{d * 100:+.1f} pts"
    if metric[4] == "int":
        return f"{d:+,.0f}"
    return f"{d:+.2f}"


def label(dim, v, private=False):
    if dim == "project":
        return "top project" if private else os.path.basename(str(v).rstrip("\\/")) or str(v)
    if v is None:
        return "none"
    return str(v)[:8] if dim in ("rules", "skills") else str(v)


def confounder_text(conf, private=False):
    name = dict(DIMS, project="project mix")
    parts = []
    for d, va, sa, vb, sb in conf:
        after = label(d, vb, private)
        if private and d == "project":
            after = "same top project" if va == vb else "a different project"
        parts.append(f"{name[d]} ({label(d, va, private)} {sa:.0%} -> {after} {sb:.0%})")
    return "Also different between the two groups: " + "; ".join(parts) if parts else ""


def tests_text(r):
    return f"intervals corrected for {r['tests']} tests (part of a {ALPHA:.0%} chance of any false alarm in the whole run)"


def change_text(row):
    m, (lo, hi) = row["metric"], row["ci"]
    return (f"{m[1]}: {fmt(m, row['before'])} -> {fmt(m, row['after'])} "
            f"({fmt_delta(m, row['after'] - row['before'])}, interval {fmt_delta(m, lo)} .. {fmt_delta(m, hi)})")


def by_split(findings):
    """Findings grouped by the version where they appeared, in first-seen order."""
    groups = {}
    for f in sorted(findings, key=lambda f: f["first_seen"]):
        groups.setdefault(f["value"], []).append(f)
    return list(groups.items())


def load(args):
    """Selected sessions, or None after saying why there are none."""
    con = open_db(args.db)
    ingest(con, args.root)
    stored = load_sessions(con)
    sessions = select(stored, args.all, args.project, getattr(args, "window", None))
    if not sessions:
        print("No sessions match these filters (try --all or another --project)." if stored else
              "No sessions found. Check --root (default: ~/.claude/projects).")
    return sessions or None


def cmd_ingest(args):
    con = open_db(args.db)
    parsed, skipped, failed, _ = ingest(con, args.root)
    total, stale = len(load_sessions(con)), stale_files(con)
    print(f"parsed {parsed} transcripts, {skipped} unchanged, {failed} unreadable; "
          f"{total} session segments stored in {args.db}")
    if stale:
        print(f"{stale} stored transcripts from an older cc-baseline are left out: their files are gone, "
              "so they cannot be re-read with the current metric definitions.")
    return 0


def cmd_report(args):
    sessions = load(args)
    if not sessions:
        return 1
    segs = segments(sessions, args.by, args.limit)
    print(f"{len(sessions)} sessions, {sessions[0]['started'][:10]} to {sessions[-1]['started'][:10]}. "
          f"By {DIMS[args.by]} (newest {len(segs)}), value +- half of the 95% interval; no interval under 5 sessions; "
          f"(n=...) marks fewer than {MIN_EVENTS} events.\n")
    w = 16
    print(f"{'':38}" + "".join(f"{label(args.by, v)[:w - 1]:>{w}}" for v, _ in segs))
    print(f"{'sessions':38}" + "".join(f"{len(g):>{w}}" for _, g in segs))
    cis = [group_cis(g) for _, g in segs]
    for j, metric in enumerate(METRICS):
        cells = []
        for (_, g), ci in zip(segs, cis):
            v, den = value(g, metric), totals(g, metric)[1]
            text = fmt(metric, v)
            if v is not None and den < MIN_EVENTS:
                text += f" (n={den})"
            elif v is not None and ci[j] is not None and ci[j][1] > ci[j][0]:
                text += " +-" + fmt(metric, (ci[j][1] - ci[j][0]) / 2).rstrip("%")
            cells.append(f" {text:>{w - 1}}")  # long cells still keep one space between columns
        print(f"{metric[1]:38}" + "".join(cells))
    return 0


def run_drift(sessions, args):
    alpha = ALPHA / (1 + len(SWITCH_DIMS))  # one share of the error budget per section
    shifts = scan(sessions, "version", args.days, alpha)
    switches = [r for r in (drift(sessions, d, args.days, alpha) for d in SWITCH_DIMS) if r]
    return shifts, switches


def print_drift_text(shifts, switches, args):
    print(f"Claude Code version: tested {shifts['tested']} version changes with {args.days}-day windows on each side "
          f"({shifts['thin']} skipped for too little data)"
          + (f", {tests_text(shifts)}." if shifts["tested"] else "."))
    for v, fs in by_split(shifts["findings"]):
        print(f"  at {v} (from {fs[0]['first_seen'][:10]}; {fs[0]['n_before']} vs {fs[0]['n_after']} sessions):")
        for f in fs:
            print(f"    {TAGS[f['worse']]:7} {change_text(f)}")
        if fs[0]["confounders"]:
            print("    " + confounder_text(fs[0]["confounders"]))
    if not shifts["findings"]:
        print("  No clear shift." if shifts["tested"] else "  Not enough data yet.")
    print()
    for r in switches:
        print(f"{DIMS[r['dim']]} {label(r['dim'], r['value'])} (first seen {r['first_seen'][:10]}): "
              f"{len(r['after'])} sessions vs {len(r['before'])} on other values in the {args.days} days before, "
              f"{tests_text(r)}")
        if r["status"] != "ok":
            print("  Not enough data yet.\n")
            continue
        changed = [x for x in r["rows"] if x["changed"]]
        for x in changed:
            print(f"  {TAGS[x['worse']]:7} {change_text(x)}")
        thin = sum(not x["enough"] for x in r["rows"])
        print(f"  {len(r['rows']) - len(changed) - thin} metrics show no clear change"
              + (f", {thin} have too few events" if thin else "") + ".")
        if r["confounders"]:
            print("  " + confounder_text(r["confounders"]))
        print()


def print_drift_md(shifts, switches, args):
    print("## cc-baseline drift report\n")
    print(f"Generated by cc-baseline {__version__} from local Claude Code transcripts. Counts only: no prompts, "
          "code, file paths or project names. Intervals use session-level bootstrap standard errors and are "
          "corrected for the number of tests; a change is listed only when its interval excludes zero.\n")
    print(f"### Claude Code version shifts\n\nTested {shifts['tested']} version changes, {args.days} days on each side"
          + (f", {tests_text(shifts)}" if shifts["tested"] else "") + f" ({shifts['thin']} skipped for too little data).\n")
    if not shifts["tested"]:
        print("Not enough data yet.")
    elif shifts["findings"]:
        print("| Metric | Before | After | Change | Interval | At version | Sessions | |")
        print("|---|---|---|---|---|---|---|---|")
        for f in shifts["findings"]:
            m, (lo, hi) = f["metric"], f["ci"]
            print(f"| {m[1]} | {fmt(m, f['before'])} | {fmt(m, f['after'])} | {fmt_delta(m, f['after'] - f['before'])} "
                  f"| {fmt_delta(m, lo)} .. {fmt_delta(m, hi)} | {f['value']} (from {f['first_seen'][:10]}) "
                  f"| {f['n_before']} / {f['n_after']} | {TAGS[f['worse']]} |")
        for v, fs in by_split(shifts["findings"]):
            if fs[0]["confounders"]:
                print(f"\nAt {v}: {confounder_text(fs[0]['confounders'], private=True)}")
    else:
        print("No clear shift.")
    for r in switches:
        print(f"\n### {DIMS[r['dim']]} {label(r['dim'], r['value'])} (first seen {r['first_seen'][:10]})\n")
        print(f"{len(r['after'])} sessions against {len(r['before'])} on other values in the {args.days} days before, "
              f"{tests_text(r)}.\n")
        if r["status"] != "ok":
            print("Not enough data yet.")
            continue
        print("| Metric | Before | After | Change | Interval | |")
        print("|---|---|---|---|---|---|")
        for x in r["rows"]:
            m = x["metric"]
            if not x["enough"]:
                print(f"| {m[1]} | {fmt(m, x['before'])} | {fmt(m, x['after'])} | | too few events | |")
                continue
            lo, hi = x["ci"]
            print(f"| {m[1]} | {fmt(m, x['before'])} | {fmt(m, x['after'])} | {fmt_delta(m, x['after'] - x['before'])} "
                  f"| {fmt_delta(m, lo)} .. {fmt_delta(m, hi)} | {TAGS[x['worse']] if x['changed'] else ''} |")
        if r["confounders"]:
            print("\n" + confounder_text(r["confounders"], private=True))


def cmd_drift(args):
    sessions = load(args)
    if not sessions:
        return 1
    (print_drift_md if args.md else print_drift_text)(*run_drift(sessions, args), args)
    return 0


def hook_message(sessions, con, deadline=None):
    """(line, notified state to save once it is shown) for worsening version shifts not reported before.

    Only version changes whose after-window has ended are tested, so each session start repeats
    the same test instead of taking a new look at growing data.
    """
    if not sessions:
        return None, None
    since = parse_ts(sessions[-1]["started"]) - timedelta(days=HOOK_LOOKBACK)
    findings = []
    for days in HOOK_WINDOWS:
        findings += scan(sessions, "version", days, ALPHA / len(HOOK_WINDOWS), since=since, iters=200,
                         complete=True, deadline=deadline)["findings"]
    seen = json.loads(get_meta(con, "notified") or "{}")  # metric key -> start of the shift already reported
    if not isinstance(seen, dict):
        seen = {}

    def is_new(f):  # the best split for one shift can move between nearby versions; report it once
        prev = seen.get(f["metric"][0])
        return prev is None or abs(parse_ts(f["first_seen"]) - parse_ts(prev)) > timedelta(days=max(HOOK_WINDOWS))

    shown = []
    for f in sorted(findings, key=lambda f: -f["strength"]):
        if f["worse"] and is_new(f) and all(f["metric"] != g["metric"] for g in shown):
            shown.append(f)
    shown = shown[:3]  # the rest wait for the next session
    if not shown:
        return None, None
    seen.update({f["metric"][0]: f["first_seen"] for f in shown})
    parts = [f"{f['metric'][1].lower()} {fmt(f['metric'], f['before'])} -> {fmt(f['metric'], f['after'])} "
             f"since {f['value']}" for f in shown]
    return ("cc-baseline: possible behaviour shift: " + "; ".join(parts) +
            f". Details and caveats: `cc-baseline drift` (CLI: {INSTALL})."), seen


def cmd_hook(args):
    """SessionStart hook: quick incremental ingest, then report only new worsening shifts. Never fails."""
    try:
        start = time.monotonic()
        sys.stdin.read()
        con = open_db(args.db)
        parsed, _, _, unfinished = ingest(con, args.root, deadline=start + HOOK_INGEST_S, skip_big=True)
        if not parsed or unfinished:  # nothing new, or a partial store that could show fake shifts
            return 0
        msg, seen = hook_message(select(load_sessions(con)), con, deadline=start + HOOK_TOTAL_S)
        if msg:
            print(json.dumps({"systemMessage": msg}), flush=True)
            set_meta(con, "notified", json.dumps(seen))  # only once the user has been told
    except Exception:  # a broken hook must never block a session
        pass
    return 0


def positive(text):
    n = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return n


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", default=default_db(), help="SQLite file (default ~/.cc-baseline/baseline.db)")
    common.add_argument("--root", default=default_root(), help="transcripts folder (default ~/.claude/projects)")
    filters = argparse.ArgumentParser(add_help=False)
    filters.add_argument("--all", action="store_true", help="include subagent and SDK/headless sessions")
    filters.add_argument("--project", help="only projects whose path contains this text")

    p = argparse.ArgumentParser(prog="cc-baseline", description="Track how Claude Code behaves across versions, "
                                "models and config, from your own transcripts.")
    p.add_argument("--version", action="version", version=f"cc-baseline {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", parents=[common], help="read new or changed transcripts into the local store")
    rep = sub.add_parser("report", parents=[common, filters], help="metrics per version, model, effort or config")
    rep.add_argument("--by", choices=DIMS, default="version")
    rep.add_argument("--limit", type=positive, default=5, help="how many newest values to show")
    rep.add_argument("--window", type=positive, metavar="DAYS", help="only the last N days of recorded sessions")
    dr = sub.add_parser("drift", parents=[common, filters], help="find version shifts and model/effort/config effects")
    dr.add_argument("--days", type=positive, default=14, help="days compared on each side of a change")
    dr.add_argument("--md", action="store_true", help="markdown you can paste into a GitHub issue")
    sub.add_parser("hook", parents=[common], help="SessionStart hook (reads the hook payload on stdin)")
    args = p.parse_args(argv)
    return {"ingest": cmd_ingest, "report": cmd_report, "drift": cmd_drift, "hook": cmd_hook}[args.cmd](args)
