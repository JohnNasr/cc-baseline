"""Behaviour metrics, bootstrap intervals and change detection over parsed sessions."""
import math
import random
import re
import time
from bisect import bisect_left
from collections import Counter
from datetime import datetime, timedelta
from statistics import NormalDist

# key, label, numerator count, denominator count, format, which direction is worse
METRICS = [
    ("read_edit", "Reads per edit", "reads", "edits", "ratio", "down"),
    ("blind_edit", "Edits without a prior Read", "edits_without_read", "edits", "pct", "up"),
    ("bash_files", "Shell used for file reads/writes", "bash_file_ops", "bash_calls", "pct", "up"),
    ("tool_errors", "Tool calls that errored", "tool_errors", "tool_calls", "pct", "up"),
    ("unverified_done", "'Done' with no check after last edit", "unverified_done", "done_claims", "pct", "up"),
    ("interrupts", "Prompts you interrupted", "interrupts", "prompts", "pct", "up"),
    ("tools_per_prompt", "Tool calls per prompt", "tool_calls", "prompts", "ratio", None),
    ("subagents_per_prompt", "Subagents per prompt", "subagents", "prompts", "ratio", None),
    ("output_per_prompt", "Output tokens per prompt", "output_tokens", "prompts", "int", None),
    ("cache_write_per_prompt", "Cache-write tokens per prompt", "cache_creation", "prompts", "int", None),
]

DIMS = {
    "version": "Claude Code version",
    "model": "model",
    "effort": "effort",
    "entrypoint": "entrypoint",
    "permission": "permission mode",
    "rules": "CLAUDE.md/rules set",
    "skills": "skill set",
}

ALPHA = 0.05  # chance of at least one false alarm across all the tests in one drift run or hook check
MIN_EVENTS = 20  # denominator events each group needs before a metric is compared
TAKEOVER = 8  # a version takes over where most of the next this-many sessions run it or newer


def parse_ts(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def version_key(v):
    return tuple(int(x) for x in re.findall(r"\d+", str(v)))


def select(sessions, include_all=False, project=None, days=None):
    """Main, interactive sessions by default; optional project substring and last-N-days filter."""
    out = []
    for s in sessions:
        if not s.get("started"):
            continue
        if not include_all and (s.get("subagent") or str(s.get("entrypoint") or "").startswith("sdk")):
            continue
        if project and project.lower() not in str(s.get("project") or "").lower():
            continue
        out.append(s)
    if days and out:
        cutoff = max(parse_ts(s["started"]) for s in out) - timedelta(days=days)
        out = [s for s in out if parse_ts(s["started"]) >= cutoff]
    return sorted(out, key=lambda s: s["started"])


def totals(group, metric):
    _, _, num, den, _, _ = metric
    return sum(s["counts"][num] for s in group), sum(s["counts"][den] for s in group)


def value(group, metric):
    num, den = totals(group, metric)
    return num / den if den else None


def _rows(group):
    return [[(s["counts"][m[2]], s["counts"][m[3]]) for m in METRICS] for s in group]


def _boot(rows, rng):
    pick = rng.choices(rows, k=len(rows))
    out = []
    for j in range(len(METRICS)):
        den = sum(p[j][1] for p in pick)
        out.append(sum(p[j][0] for p in pick) / den if den else None)
    return out


def _draws(rows, iters, seed):
    rng = random.Random(seed)
    return [_boot(rows, rng) for _ in range(iters)]


def _usable(samples):
    vals = [v for v in samples if v is not None]
    return vals if vals and len(vals) >= 0.9 * len(samples) else None


def group_cis(group, level=0.95, iters=1000, seed=0, min_sessions=5):
    """Percentile bootstrap interval for every metric of one group (None when too few sessions)."""
    if len(group) < min_sessions:  # resampling 1-4 sessions gives a falsely narrow interval
        return [None] * len(METRICS)
    draws, out = _draws(_rows(group), iters, seed), []
    for j in range(len(METRICS)):
        vals = _usable([d[j] for d in draws])
        if vals is None:
            out.append(None)
            continue
        vals.sort()
        out.append((vals[int((1 - level) / 2 * (len(vals) - 1))], vals[int((1 + level) / 2 * (len(vals) - 1))]))
    return out


def _variance(vals):
    mean = sum(vals) / len(vals)
    return sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)


def t_quantile(p, df):
    """Student t quantile via its expansion around the normal one (Abramowitz & Stegun 26.7.5)."""
    z = NormalDist().inv_cdf(p)
    terms = ((z ** 3 + z) / 4, (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96,
             (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384,
             (79 * z ** 9 + 776 * z ** 7 + 1482 * z ** 5 - 1920 * z ** 3 - 945 * z) / 92160)
    return z + sum(g / df ** (k + 1) for k, g in enumerate(terms))


def permutation_agrees(a, b, j, threshold, seed=0):
    """True when shuffling sessions between the groups gives a difference this large with p <= threshold.

    Makes no assumption about the shape of per-session counts. Stops as soon as the
    p-value can no longer get under `threshold`, so a clear non-change costs few shuffles.
    """
    pairs = [r[j] for r in _rows(a) + _rows(b)]
    n_a, rng = len(a), random.Random(seed)
    total_num, total_den = sum(p[0] for p in pairs), sum(p[1] for p in pairs)
    observed = abs(value(b, METRICS[j]) - value(a, METRICS[j]))
    iters = math.ceil(5 / threshold)
    allowed = threshold * (iters + 1) - 1  # p = (hits + 1) / (iters + 1) stays <= threshold up to here
    hits = 0
    for _ in range(iters):
        rng.shuffle(pairs)
        num_a, den_a = sum(p[0] for p in pairs[:n_a]), sum(p[1] for p in pairs[:n_a])
        den_b = total_den - den_a
        if not den_a or not den_b or abs((total_num - num_a) / den_b - num_a / den_a) >= observed - 1e-12:
            hits += 1
            if hits > allowed:
                return False
    return True


def compare(a, b, alpha=ALPHA, tests=len(METRICS), min_events=MIN_EVENTS, iters=500, seed=0, confirm=True):
    """One row per metric: before/after values, interval of the difference, and a verdict.

    The standard error comes from resampling sessions in each group. It never drops below
    the binomial (pct) or Poisson (other metrics) error at the pooled rate, so a group with
    no events at all cannot look certain. The interval uses a t value with Welch degrees of
    freedom and is Bonferroni-corrected for `tests`. Far in the tail, where that correction
    puts the cut-off, skewed counts make any such interval too narrow, so with `confirm` a
    change must also pass a permutation test at the same level.
    """
    threshold = alpha / max(tests, 1)
    rows_a, rows_b = _rows(a), _rows(b)
    draws_a, draws_b = _draws(rows_a, iters, seed), _draws(rows_b, iters, seed + 1)
    rows = []
    for j, metric in enumerate(METRICS):
        (num_a, den_a), (num_b, den_b) = totals(a, metric), totals(b, metric)
        va, vb = value(a, metric), value(b, metric)
        sa, sb = _usable([d[j] for d in draws_a]), _usable([d[j] for d in draws_b])
        row = {"metric": metric, "before": va, "after": vb, "ci": None, "enough": False, "changed": False,
               "worse": None}
        rows.append(row)
        if va is None or vb is None or sa is None or sb is None or min(den_a, den_b) < min_events or min(len(a), len(b)) < 2:
            continue
        pooled = (num_a + num_b) / (den_a + den_b)
        unit = pooled * (1 - pooled) if metric[4] == "pct" else pooled
        # n/(n-1): the bootstrap variance of a mean runs low by that factor
        var_a = max(_variance(sa) * len(a) / (len(a) - 1), unit / den_a)
        var_b = max(_variance(sb) * len(b) / (len(b) - 1), unit / den_b)
        se = math.sqrt(var_a + var_b)
        df = (var_a + var_b) ** 2 / (var_a ** 2 / (len(a) - 1) + var_b ** 2 / (len(b) - 1)) if se else 1
        half = t_quantile(1 - threshold / 2, max(df, 1)) * se
        diff = vb - va
        lo, hi = diff - half, diff + half
        changed = se > 0 and (lo > 0 or hi < 0)
        if changed and confirm:
            changed = permutation_agrees(a, b, j, threshold, seed + 2 + j)
        row.update(ci=(lo, hi), enough=True, changed=changed,
                   worse=(diff > 0) == (metric[5] == "up") if changed and metric[5] else None)
    return rows


def first_seen(sessions, dim):
    first = {}
    for s in sessions:
        if s.get(dim):
            first.setdefault(s[dim], s["started"])
    return first


def splits(sessions, dim):
    """(value, start) for each point where a new value of `dim` came in, oldest first. Sessions sorted by start.

    For versions, only releases newer than every version seen before count (opening an old
    install late is not a change), and the point is where the release takes over: the first
    session from which most of the next TAKEOVER sessions run it or newer. A trial session
    days before the switch is not where behaviour changed.
    """
    first = sorted(first_seen(sessions, dim).items(), key=lambda kv: kv[1])
    if dim != "version":
        return first[1:]  # the first value has nothing before it
    order = [s for s in sessions if s.get("version")]
    keys = [version_key(s["version"]) for s in order]
    out, top, start = [], None, 0
    for v, ts in first:
        k = version_key(v)
        if top is not None and k <= top:
            continue
        top = k
        i = max(start, next(n for n, s in enumerate(order) if s["version"] == v))
        while i < len(order) and keys[i] < k or 2 * sum(x >= k for x in keys[i:i + TAKEOVER]) <= len(keys[i:i + TAKEOVER]):
            i += 1
            if i >= len(order):
                break
        if i >= len(order):
            continue  # never adopted
        start = i
        if out and out[-1][1] == order[i]["started"]:
            out.pop()  # two releases took over at the same point: name it after the newer one
        out.append((v, order[i]["started"]))
    return out[1:]


def confounders(a, b, dim, gap=0.25):
    """Other fingerprint fields (and the project mix) whose mix differs between the groups."""
    out = []
    for d in [*DIMS, "project"]:
        if d == dim:
            continue
        ca, cb = Counter(s.get(d) for s in a), Counter(s.get(d) for s in b)
        if max(abs(ca[v] / len(a) - cb[v] / len(b)) for v in set(ca) | set(cb)) > gap:
            (top_a, n_a), (top_b, n_b) = ca.most_common(1)[0], cb.most_common(1)[0]
            out.append((d, top_a, n_a / len(a), top_b, n_b / len(b)))
    return out


def drift(sessions, dim, days=30, alpha=ALPHA, min_sessions=5, min_events=MIN_EVENTS, iters=500):
    """Sessions on the newest value of `dim` against other values in the `days` before it appeared.

    Suits fields you switch back and forth (model, effort, rules, skills).
    """
    first = first_seen(sessions, dim)
    if len(first) < 2:
        return None
    new = max(first, key=first.get)
    t0 = parse_ts(first[new])
    b = [s for s in sessions if s.get(dim) == new]
    a = [s for s in sessions
         if s.get(dim) and s.get(dim) != new and t0 - timedelta(days=days) <= parse_ts(s["started"]) < t0]
    result = {"dim": dim, "value": new, "first_seen": first[new], "before": a, "after": b, "alpha": alpha,
              "tests": len(METRICS), "rows": [], "status": "not enough data", "confounders": []}
    if len(a) >= min_sessions and len(b) >= min_sessions:
        result.update(status="ok", confounders=confounders(a, b, dim),
                      rows=compare(a, b, alpha, len(METRICS), min_events, iters))
    return result


def scan(sessions, dim="version", days=14, alpha=ALPHA, since=None, min_sessions=8, min_events=MIN_EVENTS,
         iters=500, complete=False, deadline=None):
    """Find where behaviour shifted: test each point where a new `dim` value came in (see splits).

    Before = sessions in the `days` before that point, after = the `days` from it (later
    versions included, since a change usually persists). Intervals are corrected for every
    split x metric in the history; `since` only skips reporting (and the cost of testing)
    older points. With `complete`, only points whose after-window has ended are tested, so a
    later run repeats the same test instead of adding a new one. For each metric and
    direction, the clearest split that also passes the permutation test wins. `deadline`
    (time.monotonic) stops testing, newest points first.
    """
    timed = sorted(((parse_ts(s["started"]), s) for s in sessions), key=lambda p: p[0])
    times, window = [t for t, _ in timed], timedelta(days=days)
    candidates, thin = [], 0
    for v, ts in splits(sessions, dim):
        t0 = parse_ts(ts)
        if complete and times[-1] < t0 + window:
            continue
        i0, i1, i2 = bisect_left(times, t0 - window), bisect_left(times, t0), bisect_left(times, t0 + window)
        if i1 - i0 < min_sessions or i2 - i1 < min_sessions:
            thin += 1
            continue
        candidates.append((v, ts, t0, i0, i1, i2))
    tests = len(candidates) * len(METRICS)
    flagged = {}  # (metric key, went up) -> [(finding, before group, after group, metric index)]
    for v, ts, t0, i0, i1, i2 in reversed(candidates):
        if since is not None and t0 < since:
            continue
        if deadline is not None and time.monotonic() > deadline:
            break
        a, b = [s for _, s in timed[i0:i1]], [s for _, s in timed[i1:i2]]
        conf = None
        for j, row in enumerate(compare(a, b, alpha, tests, min_events, iters, confirm=False)):
            if not row["changed"]:
                continue
            lo, hi = row["ci"]
            conf = confounders(a, b, dim) if conf is None else conf
            f = dict(row, value=v, first_seen=ts, strength=abs(row["after"] - row["before"]) / (hi - lo),
                     n_before=len(a), n_after=len(b), confounders=conf)
            flagged.setdefault((row["metric"][0], row["after"] > row["before"]), []).append((f, a, b, j))
    findings = []
    for options in flagged.values():
        for f, a, b, j in sorted(options, key=lambda o: -o[0]["strength"]):
            if deadline is not None and time.monotonic() > deadline:
                break
            if permutation_agrees(a, b, j, alpha / tests, seed=j):
                findings.append(f)
                break
    findings.sort(key=lambda f: -f["strength"])
    return {"dim": dim, "tested": len(candidates), "thin": thin, "days": days, "alpha": alpha, "tests": tests,
            "findings": findings}


def segments(sessions, dim, limit=5):
    """The newest `limit` values of `dim`, each with its sessions, oldest first (versions by number)."""
    first = first_seen(sessions, dim)
    order = sorted(first, key=version_key if dim == "version" else first.get)
    return [(v, [s for s in sessions if s.get(dim) == v]) for v in order[-limit:]] if limit > 0 else []
