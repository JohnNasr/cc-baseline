import io
import json
import os
from datetime import datetime, timedelta

import pytest

from ccbaseline import cli
from ccbaseline.parse import COUNT_KEYS, is_file_op, parse_file
from ccbaseline.stats import METRICS, compare, confounders, drift, scan, segments, select, splits, t_quantile
from ccbaseline.store import ingest, load_sessions, open_db, stale_files

PROJECT = "/home/u/secret-client-project"


class Transcript:
    """Builds a Claude Code-style JSONL transcript line by line."""

    def __init__(self, version="2.1.0", day=1, session="s1"):
        self.lines, self.version, self.day, self.session, self.minute = [], version, day, session, 0

    def _rec(self, kind, **kw):
        self.minute += 1
        r = {"type": kind, "timestamp": f"2026-01-{self.day:02d}T10:{self.minute:02d}:00.000Z",
             "sessionId": self.session, "cwd": PROJECT, "version": self.version, "entrypoint": "cli"}
        r.update(kw)
        self.lines.append(r)
        return self

    def prompt(self, text="please fix it"):
        return self._rec("user", permissionMode="default", message={"role": "user", "content": text})

    def interrupt(self):
        return self._rec("user", message={"role": "user", "content": [
            {"type": "text", "text": "[Request interrupted by user for tool use]"}]})

    def say(self, mid, text, usage=None):
        return self._rec("assistant", effort="high", message={
            "id": mid, "model": "claude-test", "usage": usage, "content": [{"type": "text", "text": text}]})

    def user(self, text, **kw):
        return self._rec("user", message={"role": "user", "content": text}, **kw)

    def attach(self, **att):
        return self._rec("attachment", attachment=att)

    def tool(self, mid, uid, name, inp, error=False, result=None, usage=None, content="ok"):
        self._rec("assistant", effort="high", message={
            "id": mid, "model": "claude-test", "usage": usage,
            "content": [{"type": "tool_use", "id": uid, "name": name, "input": inp}]})
        extra = {"toolUseResult": result} if result else {}
        return self._rec("user", message={"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": uid, "is_error": error, "content": content}]}, **extra)

    def write(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("".join(json.dumps(r) + "\n" for r in self.lines))
        return path


def rich_transcript():
    usage = {"input_tokens": 5, "output_tokens": 10, "cache_creation_input_tokens": 100, "cache_read_input_tokens": 50}
    t = Transcript()
    t._rec("attachment", attachment={"type": "instructions", "files": [{"path": "CLAUDE.md", "content": "be brief"}]})
    t._rec("attachment", attachment={"type": "skill_listing", "content": "- skill-a"})
    t.prompt()
    t.say("m1", "Let me look.", usage)  # same message id streamed over two lines: usage must count once
    t.tool("m1", "r1", "Read", {"file_path": "/p/a.py"}, usage=usage)
    t.tool("m2", "e1", "Edit", {"file_path": "/p/a.py"})
    t.tool("m3", "e2", "Edit", {"file_path": "/p/b.py"})  # never read: blind edit
    t.tool("m4", "b1", "Bash", {"command": "cat /p/c.py"})
    t.tool("m5", "b2", "Bash", {"command": "pytest -q"})
    t.say("m6", "Done, tests pass.")  # verified: the check ran after the last edit
    t.prompt("now add a feature")
    t.tool("m7", "w1", "Write", {"file_path": "/p/new.py"}, result={"type": "create"})
    t.tool("m8", "e3", "Edit", {"file_path": "/p/zzz.py"}, error=True)  # refused edit
    t.say("m9", "Implemented.")  # not verified: no check after the Write
    t.interrupt()
    t._rec("system", subtype="compact_boundary")
    t.tool("m10", "a1", "Agent", {"prompt": "x"})
    t.version = "2.2.0"
    t.prompt("after the update")
    return t


def test_parse_counts_and_segments(tmp_path):
    segs = parse_file(rich_transcript().write(str(tmp_path / "proj" / "s1.jsonl")))
    assert [s["version"] for s in segs] == ["2.1.0", "2.2.0"]
    c = segs[0]["counts"]
    expected = dict(prompts=2, interrupts=1, tool_calls=8, tool_errors=1, reads=1, edits=3, creates=1,
                    edits_without_read=2, bash_calls=2, bash_file_ops=1, subagents=1, compactions=1,
                    done_claims=2, unverified_done=1, input_tokens=5, cache_creation=100, cache_read=50,
                    output_tokens=10)
    assert c == expected
    assert set(c) == set(COUNT_KEYS)
    s = segs[0]
    assert (s["model"], s["effort"], s["entrypoint"], s["permission"]) == ("claude-test", "high", "cli", "default")
    assert s["project"] == PROJECT and s["rules"] and s["skills"] and not s["subagent"]
    assert segs[1]["counts"]["prompts"] == 1 and segs[1]["rules"] == s["rules"]


def test_rules_hash_changes_with_content(tmp_path):
    def rules_of(content):
        t = Transcript()
        t._rec("attachment", attachment={"type": "instructions", "files": [{"path": "CLAUDE.md", "content": content}]})
        t.prompt()
        return parse_file(t.write(str(tmp_path / f"{content}.jsonl")))[0]["rules"]
    assert rules_of("a") != rules_of("b")


def test_subagent_and_empty_files(tmp_path):
    t = Transcript().prompt()
    assert parse_file(t.write(str(tmp_path / "p" / "s" / "subagents" / "agent-1.jsonl")))[0]["subagent"]
    junk = tmp_path / "junk.jsonl"
    junk.write_text("not json\n[1, 2]\n", encoding="utf-8")
    assert parse_file(str(junk)) == []


@pytest.mark.parametrize("command,expected", [
    ("cat a.txt", True), ("git status", False), ("FOO=1 grep -r x .", True), ("echo hi > out.txt", True),
    ("echo hi 2>&1", False), ("ls && sed -n 1p f", True), ("Get-Content x.txt", True),
    ("npm test | grep fail", False), ("/usr/bin/head -5 f", True), ("", False),
])
def test_is_file_op(command, expected):
    assert is_file_op(command) is expected


def test_ingest_is_incremental(tmp_path):
    root, db = tmp_path / "projects", str(tmp_path / "b.db")
    path = Transcript().prompt().write(str(root / "p" / "s1.jsonl"))
    con = open_db(db)
    assert ingest(con, str(root))[:3] == (1, 0, 0)
    assert ingest(con, str(root))[:3] == (0, 1, 0)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(Transcript().prompt().lines[0]) + "\n")
    assert ingest(con, str(root))[:3] == (1, 0, 0)
    assert load_sessions(con)[0]["counts"]["prompts"] == 2
    os.remove(path)  # Claude Code prunes old transcripts; the store keeps them
    assert len(load_sessions(con)) == 1


def test_ingest_keys_files_once_and_escapes_glob(tmp_path):
    root = tmp_path / "cfg [old]" / "projects"
    Transcript().prompt().write(str(root / "p" / "s1.jsonl"))
    con = open_db(str(tmp_path / "b.db"))
    assert ingest(con, str(root))[:3] == (1, 0, 0)
    assert ingest(con, str(root).replace(os.sep, "/") + "/")[:3] == (0, 1, 0)
    assert len(load_sessions(con)) == 1


def session(day, version="v1", model="m", blind=0, edits=10, project="p"):
    counts = dict.fromkeys(COUNT_KEYS, 0)
    counts.update(prompts=5, reads=10, edits=edits, edits_without_read=blind, tool_calls=20)
    started = (datetime(2026, 1, 1, 10) + timedelta(days=day - 1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"session_id": f"{version}-{day}", "started": started, "version": version,
            "model": model, "effort": "high", "entrypoint": "cli", "permission": "default", "rules": "r",
            "skills": "s", "project": project, "subagent": False, "counts": counts}


def counts_of(t, tmp_path, name="s.jsonl"):
    return parse_file(t.write(str(tmp_path / name)))[0]["counts"]


def test_prompt_counting_skips_commands_output_and_notifications(tmp_path):
    t = Transcript().prompt("real question")
    t.say("m1", "answer")
    t.user("<command-name>/model</command-name>")  # local command: no model reply
    t.user("<local-command-stdout>Set model</local-command-stdout>")
    t.user("<command-message>review</command-message><command-name>/review</command-name>")  # answered: a prompt
    t.user("expanded review instructions", isMeta=True)
    t.say("m2", "reviewing")
    t.user("<task-notification>task done</task-notification>", origin={"kind": "task-notification"})
    t.user("<system-reminder>note</system-reminder>")
    t.user("<bash-input>ls</bash-input>")
    t.user("<bash-stdout>a b</bash-stdout>")
    t.attach(type="queued_command", commandMode="prompt", origin={"kind": "human"}, prompt="typed mid-turn")
    t.attach(type="queued_command", commandMode="prompt", origin={"kind": "peer"}, prompt="from a peer")
    t.attach(type="queued_command", commandMode="task-notification", prompt="x")
    assert counts_of(t, tmp_path)["prompts"] == 3


def test_command_only_session_is_ignored(tmp_path):
    t = Transcript().user("<command-name>/model</command-name>").user("<local-command-stdout>ok</local-command-stdout>")
    assert parse_file(t.write(str(tmp_path / "c.jsonl"))) == []


def test_skills_hash_ignores_delta_listings(tmp_path):
    def skills_of(*listings):
        t = Transcript()
        for content, initial in listings:
            t.attach(type="skill_listing", content=content, isInitial=initial)
        t.prompt()
        return parse_file(t.write(str(tmp_path / f"{len(listings)}.jsonl")))[0]["skills"]
    assert skills_of(("a b c", True), ("d", False)) == skills_of(("a b c", True))


def test_forked_history_is_not_counted_twice(tmp_path):
    own = "11111111-2222-3333-4444-555555555555"
    parent = Transcript(session="99999999-2222-3333-4444-555555555555").prompt("old").prompt("older")
    t = Transcript(session=own)
    t.lines = [dict(r) for r in parent.lines]  # copied from the parent's file
    t.prompt("new question")
    seg = parse_file(t.write(str(tmp_path / f"{own}.jsonl")))
    assert len(seg) == 1 and seg[0]["counts"]["prompts"] == 1 and seg[0]["session_id"] == own


def test_subagent_creates_paths_and_refusals(tmp_path):
    t = Transcript().prompt()
    t.tool("m1", "w1", "Write", {"file_path": "/p/new.py"}, content="File created successfully at: /p/new.py")
    t.tool("m2", "r1", "Read", {"file_path": r"C:\p\a.py"})
    t.tool("m3", "e1", "Edit", {"file_path": "/c/p/a.py"})  # same file, Git Bash spelling
    t.tool("m4", "w2", "Write", {"file_path": "/p/old.py"}, error=True,
           content="<tool_use_error>File has not been read yet. Read it first</tool_use_error>")
    t.tool("m5", "w3", "Write", {"file_path": "/p/x.py"}, error=True, content="denied")  # ignored
    c = counts_of(t, tmp_path)
    assert (c["creates"], c["edits"], c["edits_without_read"]) == (1, 2, 1)


def test_forked_subagent_task_counts_as_prompt(tmp_path):
    t = Transcript()
    t._rec("user", message={"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "x", "content": "context"},
        {"type": "text", "text": "<fork-boilerplate>do the task</fork-boilerplate>"}]})
    t.tool("m1", "r1", "Read", {"file_path": "/p/a.py"})
    seg = parse_file(t.write(str(tmp_path / "p" / "s" / "subagents" / "agent-f.jsonl")))
    assert seg and seg[0]["counts"]["prompts"] == 1


def test_scan_finds_a_clear_shift():
    sessions = [session(d, "v1", blind=0) for d in range(1, 15)] + [session(d, "v2", blind=5) for d in range(15, 29)]
    result = scan(sessions)
    assert result["tested"] == 1
    keys = {f["metric"][0]: f for f in result["findings"]}
    assert set(keys) == {"blind_edit"}
    f = keys["blind_edit"]
    assert f["value"] == "v2" and f["worse"] is True and f["before"] == 0 and f["after"] == 0.5
    assert f["confounders"] == []


def test_scan_quiet_without_change_or_data():
    same = [session(d, "v1") for d in range(1, 15)] + [session(d, "v2") for d in range(15, 29)]
    assert scan(same)["findings"] == []
    thin = [session(d, "v1") for d in range(1, 4)] + [session(d, "v2") for d in range(4, 7)]
    assert scan(thin)["tested"] == 0 and scan(thin)["thin"] == 1


def test_compare_does_not_trust_a_group_with_no_events():
    # 0/40 blind edits before, 8/150 after: Fisher's exact test gives p = 0.2, so this is not a change
    a = [session(d, edits=5) for d in range(1, 9)]
    b = [session(d, edits=15, blind=b) for d, b in zip(range(10, 20), [1, 1, 1, 1, 1, 1, 1, 1, 0, 0])]
    row = next(r for r in compare(a, b) if r["metric"][0] == "blind_edit")
    assert row["enough"] and not row["changed"]


@pytest.mark.parametrize("p, df, want", [(0.9975, 27, 3.057), (0.975, 10, 2.228), (0.995, 4, 4.604)])
def test_t_quantile_matches_tables(p, df, want):
    assert t_quantile(p, df) == pytest.approx(want, abs=0.015)


def test_old_versions_seen_late_are_not_splits():
    sessions = [session(1, "2.1.10"), session(2, "2.1.12"), session(3, "2.1.11"), session(4, "2.1.13")]
    assert [v for v, _ in splits(sessions, "version")] == ["2.1.12", "2.1.13"]
    assert [v for v, _ in segments(sessions, "version", 3)] == ["2.1.11", "2.1.12", "2.1.13"]


def test_confounders_compare_mixes_not_top_labels():
    a = [session(d, model=m) for d, m in zip(range(1, 9), "XYXYXYXY")]
    b = [session(d, model=m) for d, m in zip(range(9, 17), "YXYXYXYX")]
    assert confounders(a, b, "version") == []
    c = [session(d, model="Z") for d in range(9, 17)]
    assert [x[0] for x in confounders(a, c, "version")] == ["model"]


def test_scan_keeps_each_direction():
    better = [session(d, "v1", blind=6) for d in range(1, 15)] + [session(d, "v2", blind=0) for d in range(15, 29)]
    worse = [session(d, "v3", blind=6) for d in range(29, 43)]
    found = {(f["metric"][0], f["worse"]): f["value"] for f in scan(better + worse)["findings"]}
    assert found == {("blind_edit", False): "v2", ("blind_edit", True): "v3"}


def test_compare_needs_enough_events():
    a, b = [session(1, edits=2)] * 5, [session(2, edits=2, blind=2)] * 5
    row = next(r for r in compare(a, b) if r["metric"][0] == "blind_edit")
    assert not row["enough"] and not row["changed"]


def test_drift_on_a_switch_lists_confounders():
    sessions = [session(d, "v1", model="a") for d in range(1, 11)] + \
               [session(d, "v2", model="b", blind=5) for d in range(11, 21)]
    r = drift(sessions, "model")
    assert r["status"] == "ok" and r["value"] == "b"
    assert [c[0] for c in r["confounders"]] == ["version"]
    assert next(x for x in r["rows"] if x["metric"][0] == "blind_edit")["worse"] is True
    few = sessions[:10] + sessions[10:12]
    assert drift(few, "model")["status"] == "not enough data"
    assert drift(sessions[:10], "model") is None  # only one model seen


def test_select_filters():
    sub = dict(session(1), subagent=True)
    sdk = dict(session(2), entrypoint="sdk-cli")
    other = session(3, project="elsewhere")
    assert select([sub, sdk, other]) == [other]
    assert len(select([sub, sdk, other], include_all=True)) == 3
    assert select([other], project="nomatch") == []


def write_shifted_history(root, days=30):
    """Ten clean sessions on 2.1.0, then twenty on 2.2.0 that edit without reading first.

    Twenty, so the hook's 14-day after-window has ended by the newest session."""
    for day in range(1, days + 1):
        blind = day > 10
        t = Transcript("2.2.0" if blind else "2.1.0", day, f"s{day}").prompt()
        for i in range(3):
            if not blind:
                t.tool(f"r{i}", f"r{i}", "Read", {"file_path": f"/p/{i}.py"})
            t.tool(f"e{i}", f"e{i}", "Edit", {"file_path": f"/p/{i}.py"})
        t.write(str(root / "p" / f"s{day}.jsonl"))


def run_cli(argv, monkeypatch, capsys, stdin="{}"):
    monkeypatch.setattr("sys.stdin", io.StringIO(stdin))
    code = cli.main(argv)
    return code, capsys.readouterr().out


def test_hook_alerts_once(tmp_path, monkeypatch, capsys):
    write_shifted_history(tmp_path / "projects")
    args = ["hook", "--db", str(tmp_path / "b.db"), "--root", str(tmp_path / "projects")]
    code, out = run_cli(args, monkeypatch, capsys)
    assert code == 0
    msg = json.loads(out)["systemMessage"]
    assert "edits without a prior read" in msg and "2.2.0" in msg
    assert run_cli(args, monkeypatch, capsys) == (0, "")  # nothing new
    t = Transcript("2.2.0", 31, "s31").prompt().tool("e", "e", "Edit", {"file_path": "/p/0.py"})
    t.write(str(tmp_path / "projects" / "p" / "s31.jsonl"))
    assert run_cli(args, monkeypatch, capsys) == (0, "")  # new data, same shift: already reported


def test_hook_never_fails(tmp_path, monkeypatch, capsys):
    bad_db = tmp_path / "is-a-folder.db"
    bad_db.mkdir()
    assert run_cli(["hook", "--db", str(bad_db), "--root", str(tmp_path)], monkeypatch, capsys) == (0, "")


def test_cli_reports_without_leaking_paths(tmp_path, monkeypatch, capsys):
    write_shifted_history(tmp_path / "projects")
    common = ["--db", str(tmp_path / "b.db"), "--root", str(tmp_path / "projects")]
    assert run_cli(["ingest", *common], monkeypatch, capsys)[0] == 0
    code, out = run_cli(["report", *common], monkeypatch, capsys)
    assert code == 0 and "Edits without a prior Read" in out
    code, out = run_cli(["drift", *common], monkeypatch, capsys)
    assert code == 0 and "worse" in out and "2.2.0" in out
    code, md = run_cli(["drift", "--md", *common], monkeypatch, capsys)
    assert code == 0 and md.startswith("## cc-baseline")
    assert "secret-client-project" not in md and "/home/u" not in md


def test_cli_empty_store(tmp_path, monkeypatch, capsys):
    common = ["--db", str(tmp_path / "b.db"), "--root", str(tmp_path)]
    assert run_cli(["report", *common], monkeypatch, capsys)[0] == 1
    assert run_cli(["drift", *common], monkeypatch, capsys)[0] == 1


def test_metric_table_is_consistent():
    for key, label, num, den, kind, worse in METRICS:
        assert num in COUNT_KEYS and den in COUNT_KEYS
        assert kind in ("pct", "ratio", "int") and worse in ("up", "down", None)


def test_reminder_first_prompt_counts_and_plugin_text_does_not(tmp_path):
    t = Transcript()
    t._rec("user", message={"role": "user", "content": [
        {"type": "text", "text": "<system-reminder>context</system-reminder>"},
        {"type": "text", "text": "real question"}]})
    t.say("m1", "answer")
    t.user("text a plugin sent", origin={"kind": "plugin"})
    t.say("m2", "ok")
    assert counts_of(t, tmp_path)["prompts"] == 1


def test_fork_keeps_parent_rules_but_starts_at_its_own_records(tmp_path):
    own = "11111111-2222-3333-4444-555555555555"
    parent = Transcript(day=1, session="99999999-2222-3333-4444-555555555555")
    parent.attach(type="instructions", files=[{"path": "CLAUDE.md", "content": "rules"}])
    parent.prompt("old")
    t = Transcript(day=5, session=own)
    t.lines = [dict(r) for r in parent.lines]
    t.prompt("new question")
    seg = parse_file(t.write(str(tmp_path / f"{own}.jsonl")))[0]
    assert seg["rules"] and seg["started"].startswith("2026-01-05") and seg["counts"]["prompts"] == 1


def test_forked_subagent_skips_the_copied_parent_prefix(tmp_path):
    t = Transcript()
    t._rec("fork-context-ref")
    t.prompt("the parent's question")
    t.tool("m0", "p0", "Edit", {"file_path": "/p/x.py"})
    t._rec("user", message={"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "p1", "content": "context"},
        {"type": "text", "text": "<fork-boilerplate>do the task</fork-boilerplate>"}]})
    t.tool("m1", "r1", "Read", {"file_path": "/p/a.py"})
    c = parse_file(t.write(str(tmp_path / "p" / "s" / "subagents" / "agent-f.jsonl")))[0]["counts"]
    assert (c["prompts"], c["edits"], c["reads"]) == (1, 0, 1)


def test_refused_unread_write_is_blind_even_after_an_earlier_read(tmp_path):
    t = Transcript().prompt()
    t.tool("m1", "r1", "Read", {"file_path": "/p/a.py"})
    t.tool("m2", "w1", "Write", {"file_path": "/p/a.py"}, error=True,
           content="<tool_use_error>File has not been read yet. Read it first</tool_use_error>")
    c = counts_of(t, tmp_path)
    assert (c["edits"], c["edits_without_read"]) == (1, 1)


def test_bad_records_are_skipped(tmp_path):
    t = Transcript().prompt()
    t._rec("assistant", message="not an object")
    t._rec("user", message="not an object")
    t.tool("m1", "r1", "Read", {"file_path": "/p/a.py"})
    c = counts_of(t, tmp_path)
    assert (c["prompts"], c["reads"]) == (1, 1)


def test_ingest_keeps_unparseable_files_and_drops_old_parser_rows(tmp_path, monkeypatch):
    root, con = tmp_path / "projects", open_db(str(tmp_path / "b.db"))
    Transcript().prompt().write(str(root / "p" / "a.jsonl"))

    def broken_parser(path):
        raise RuntimeError("parser bug")
    monkeypatch.setattr("ccbaseline.store.parse_file", broken_parser)
    assert ingest(con, str(root))[:3] == (0, 0, 1)
    assert ingest(con, str(root))[:3] == (0, 1, 0)  # not retried until the file changes
    con.execute("INSERT INTO files VALUES ('gone', 0, 0, 1, ?)", (json.dumps([session(1)]),))
    assert load_sessions(con) == [] and stale_files(con) == 1


def test_version_split_is_where_the_new_version_takes_over():
    days = [(d, "v1") for d in range(1, 12) if d != 5] + [(5, "v2")] + [(d, "v2") for d in range(12, 20)]
    sessions = sorted((session(d, v) for d, v in days), key=lambda s: s["started"])
    assert splits(sessions, "version") == [("v2", session(12)["started"])]


def test_one_outlier_session_is_not_a_shift():
    a = [session(d) for d in range(1, 11)]
    b = [session(d) for d in range(11, 20)]
    big = session(20)
    big["counts"] = dict(big["counts"], tool_calls=400)
    row = next(r for r in compare(a, b + [big]) if r["metric"][0] == "tools_per_prompt")
    assert row["enough"] and not row["changed"]


def test_hook_waits_until_the_after_window_has_ended(tmp_path, monkeypatch, capsys):
    write_shifted_history(tmp_path / "projects", days=20)
    common = ["--db", str(tmp_path / "b.db"), "--root", str(tmp_path / "projects")]
    assert run_cli(["hook", *common], monkeypatch, capsys) == (0, "")
    code, out = run_cli(["drift", *common], monkeypatch, capsys)
    assert code == 0 and "2.2.0" in out  # the full drift run does test it


def test_cli_filters_limits_and_thin_markdown(tmp_path, monkeypatch, capsys):
    write_shifted_history(tmp_path / "projects", days=3)
    common = ["--db", str(tmp_path / "b.db"), "--root", str(tmp_path / "projects")]
    code, out = run_cli(["report", "--project", "nomatch", *common], monkeypatch, capsys)
    assert code == 1 and "match these filters" in out
    code, md = run_cli(["drift", "--md", *common], monkeypatch, capsys)
    assert code == 0 and "Not enough data yet." in md and "corrected for 0" not in md
    with pytest.raises(SystemExit):
        cli.main(["report", "--limit", "0", *common])
