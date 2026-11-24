"""Turn one Claude Code transcript (.jsonl) into fingerprints plus behaviour counts.

A transcript resumed after an update spans several Claude Code versions, so it is split
into one segment per version. Only counts and hashes leave this module: no prompt text,
code or file contents.
"""
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone

PARSER_VERSION = 5

COUNT_KEYS = (
    "prompts", "interrupts", "tool_calls", "tool_errors", "reads", "edits", "creates",
    "edits_without_read", "bash_calls", "bash_file_ops", "subagents", "compactions",
    "done_claims", "unverified_done", "input_tokens", "cache_creation", "cache_read",
    "output_tokens",
)

EDIT_TOOLS = {"Edit", "MultiEdit", "NotebookEdit"}
SHELL_TOOLS = {"Bash", "PowerShell"}
AGENT_TOOLS = {"Agent", "Task"}

# First word of a shell segment that does a job Read/Grep/Glob/Edit/Write exist for.
FILE_CMDS = {
    "cat", "head", "tail", "less", "more", "sed", "awk", "grep", "egrep", "fgrep", "rg", "find",
    "get-content", "gc", "select-string", "sls", "set-content", "add-content", "out-file",
}
REDIRECT_WRITE = re.compile(r"^(echo|printf)\b.*[^0-9&]>{1,2}\s*[\w./~\\-]")
CHECK_CMD = re.compile(
    r"\b(pytest|py\.test|unittest|vitest|jest|mocha|playwright|go (test|vet|build)|"
    r"cargo (test|check|clippy|build)|(npm|pnpm|yarn|bun)( run)? (test|lint|build|check|typecheck)|"
    r"bun test|deno (test|check)|tsc|eslint|ruff|mypy|pyright|flake8|node --check|"
    r"dotnet (test|build)|mvn|gradlew?|make (test|check)|phpunit|rspec|swift (test|build)|ctest)\b"
)
DONE_CLAIM = re.compile(
    r"\b(done|fixed|works( now)?|working now|all (tests )?pass(es|ing)?|tests? pass(es|ing)?|"
    r"completed?|finished|implemented|ready)\b",
    re.IGNORECASE,
)
INTERRUPT = "[Request interrupted by user"
REMINDER = ("<system-reminder>", "</system-reminder>")
# User records that are not something the person asked: command output, notifications, reminders.
NOT_PROMPT = ("<task-notification>", "<local-command-stdout>", "<local-command-stderr>", REMINDER[0],
              "<bash-input>", "<bash-stdout>", "<bash-stderr>")
COMMAND = ("<command-name>", "<command-message>")  # a prompt only when the model answers it
FORK = "<fork-boilerplate>"  # a forked subagent's task arrives next to a tool_result
UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
# A record in a shape this parser does not expect is skipped, not allowed to drop the whole file.
BAD_RECORD = (AttributeError, TypeError, KeyError, ValueError, RecursionError)


def _norm_path(p):
    p = str(p or "").replace("\\", "/").lower()
    return re.sub(r"^/([a-z])/", r"\1:/", p)  # Git Bash /c/x and C:\x name the same file


def is_file_op(command):
    """True when a shell command reads, searches or writes files directly."""
    for segment in re.split(r"&&|\|\||;|\n", str(command or "")):
        first = segment.split("|")[0].strip()
        words = [w for w in first.split() if not re.match(r"^\w+=", w)]  # drop FOO=bar prefixes
        if not words:
            continue
        cmd = os.path.basename(words[0].strip("'\"")).lower().removesuffix(".exe")
        if cmd in FILE_CMDS or REDIRECT_WRITE.match(" ".join(words)):
            return True
    return False


def _hash(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()[:10]


def _str(v):
    return v if isinstance(v, str) and v else None


def _int(v):
    return v if isinstance(v, int) and not isinstance(v, bool) else 0


def _ts(v):
    """An ISO timestamp in UTC 'Z' form, or None when it is missing, malformed or has no time zone."""
    if not isinstance(v, str):
        return None
    try:
        dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        return None
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _texts(record):
    """Text blocks of a user record (leaving out blocks that are only a system reminder) and whether it holds a tool result."""
    content = (record.get("message") or {}).get("content")
    if isinstance(content, str):
        texts, has_result = [content], False
    elif isinstance(content, list):
        texts = [b.get("text") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        has_result = any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
    else:
        return [], False
    reminder = lambda t: t.lstrip().startswith(REMINDER[0]) and t.rstrip().endswith(REMINDER[1])
    return [t for t in texts if isinstance(t, str) and not reminder(t)], has_result


def _result_text(block):
    content = block.get("content")
    if isinstance(content, list):
        return " ".join(str(b.get("text", "")) for b in content if isinstance(b, dict))
    return str(content or "")


def _kind(origin):
    return origin.get("kind") if isinstance(origin, dict) else origin


class _Segment:
    def __init__(self, version=None):
        self.version = version
        self.counts = dict.fromkeys(COUNT_KEYS, 0)
        self.fp = {k: Counter() for k in ("model", "effort", "entrypoint", "permission")}
        self.usage = {}  # message id -> usage (streamed lines repeat it; keep the last)
        self.first_ts = self.last_ts = None

    def result(self, base):
        if not self.counts["prompts"]:
            return None
        for u in self.usage.values():
            self.counts["input_tokens"] += _int(u.get("input_tokens"))
            self.counts["cache_creation"] += _int(u.get("cache_creation_input_tokens"))
            self.counts["cache_read"] += _int(u.get("cache_read_input_tokens"))
            self.counts["output_tokens"] += _int(u.get("output_tokens"))
        out = {k: (v.most_common(1)[0][0] if v else None) for k, v in self.fp.items()}
        out.update(base, version=self.version, started=self.first_ts, ended=self.last_ts, counts=self.counts)
        return out


class _Turn:
    def __init__(self):
        self.changed = False
        self.last_edit = -1
        self.last_check = -1
        self.last_text = ""


def parse_file(path):
    """Return a list of session segments (one per Claude Code version) that contain prompts."""
    subagent = "/subagents/" in _norm_path(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    own = stem if UUID.match(stem) and not subagent else None  # main transcripts are named <session id>.jsonl
    base = {"session_id": own, "project": None, "rules": None, "skills": None, "subagent": subagent}
    pending = {}  # tool_use id -> (name, input)
    read_paths = set()
    done = []
    seg = _Segment()
    turn = _Turn()
    seq = 0
    awaiting_reply = False  # a slash command was typed; it is a prompt if the model answers
    fork_prefix = False  # a forked subagent file starts with the parent's copied spawning message

    def close_turn():
        if turn.changed and DONE_CLAIM.search(turn.last_text):
            seg.counts["done_claims"] += 1
            seg.counts["unverified_done"] += turn.last_check < turn.last_edit

    def context(att):
        if att.get("type") == "instructions":
            base["rules"] = _hash([[f.get("path"), f.get("content")] for f in att.get("files") or []])
        elif att.get("type") == "skill_listing" and att.get("isInitial", True):  # later ones are deltas
            base["skills"] = _hash(att.get("content"))

    def handle(r):
        nonlocal seg, turn, seq, awaiting_reply, fork_prefix
        kind = r.get("type")
        if kind == "fork-context-ref":
            fork_prefix = True
            return
        if fork_prefix:  # skip the copied parent records up to the fork's own task
            texts = _texts(r)[0] if kind == "user" else []
            if not (texts and texts[0].lstrip().startswith(FORK)):
                return
            fork_prefix = False
        sid = r.get("sessionId")
        if own and sid not in (None, own):  # history copied from the parent when this session was forked
            if kind == "attachment" and isinstance(r.get("attachment"), dict):
                context(r["attachment"])  # the fork still saw the parent's rules and skills
            return
        base["session_id"] = base["session_id"] or _str(sid)

        version = _str(r.get("version"))
        if kind in ("user", "assistant") and version and version != seg.version:
            if seg.version is not None:  # resumed on another version: start a new segment
                close_turn()
                done.append(seg.result(dict(base)))
                seg, turn = _Segment(), _Turn()
            seg.version = version
        ts = _ts(r.get("timestamp"))
        if ts and (own is None or sid == own):
            seg.first_ts = min(seg.first_ts or ts, ts)
            seg.last_ts = max(seg.last_ts or ts, ts)
        if kind in ("user", "assistant"):
            base["project"] = base["project"] or _str(r.get("cwd"))
            if _str(r.get("entrypoint")):
                seg.fp["entrypoint"][r["entrypoint"]] += 1
        if kind == "permission-mode" and _str(r.get("permissionMode")):
            seg.fp["permission"][r["permissionMode"]] += 1

        if kind == "assistant":
            msg = r.get("message") or {}
            model = _str(msg.get("model"))
            if model and model != "<synthetic>":
                seg.fp["model"][model] += 1
                if awaiting_reply:
                    seg.counts["prompts"] += 1
                    awaiting_reply = False
            if _str(r.get("effort")):
                seg.fp["effort"][r["effort"]] += 1
            if _str(msg.get("id")) and isinstance(msg.get("usage"), dict):
                seg.usage[msg["id"]] = msg["usage"]
            content = msg.get("content")
            for block in content if isinstance(content, list) else []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    seg.counts["tool_calls"] += 1
                    inp = block.get("input")
                    pending[block.get("id")] = (block.get("name"), inp if isinstance(inp, dict) else {})
                    if block.get("name") in AGENT_TOOLS:
                        seg.counts["subagents"] += 1
                elif block.get("type") == "text" and _str(block.get("text")):
                    turn.last_text = block["text"]

        elif kind == "user":
            if _str(r.get("permissionMode")):
                seg.fp["permission"][r["permissionMode"]] += 1
            content = (r.get("message") or {}).get("content")
            for block in content if isinstance(content, list) else []:
                if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                    continue
                name, inp = pending.pop(block.get("tool_use_id"), (None, {}))
                if name is None:
                    continue
                seq += 1
                error = bool(block.get("is_error"))
                seg.counts["tool_errors"] += error
                target = _norm_path(inp.get("file_path") or inp.get("notebook_path"))
                result = r.get("toolUseResult") if isinstance(r.get("toolUseResult"), dict) else {}
                text = _result_text(block)
                # subagent results carry no toolUseResult, so fall back to the result text
                created = result.get("type") == "create" or text.startswith("File created")
                unread = error and "not been read" in text  # Claude Code itself refused: the file was not read
                if name == "Read":
                    seg.counts["reads"] += 1
                    if not error:
                        read_paths.add(target)
                elif name in EDIT_TOOLS or (name == "Write" and not created and (not error or unread)):
                    seg.counts["edits"] += 1
                    seg.counts["edits_without_read"] += unread or target not in read_paths
                    if not error:  # a refused edit neither reads nor changes the file
                        read_paths.add(target)
                        turn.changed, turn.last_edit = True, seq
                elif name == "Write" and not error:
                    seg.counts["creates"] += 1
                    read_paths.add(target)
                    turn.changed, turn.last_edit = True, seq
                elif name in SHELL_TOOLS:
                    command = inp.get("command", "")
                    seg.counts["bash_calls"] += 1
                    seg.counts["bash_file_ops"] += is_file_op(command)
                    if not error and CHECK_CMD.search(str(command)):
                        turn.last_check = seq
            texts, has_result = _texts(r)
            first = texts[0].lstrip() if texts else ""
            meta = r.get("isMeta") or r.get("isCompactSummary") or r.get("isVisibleInTranscriptOnly")
            if first and not meta:
                awaiting_reply = False
            if any(t.startswith(INTERRUPT) for t in texts):
                seg.counts["interrupts"] += 1
            elif not first or meta or first.startswith(NOT_PROMPT) or _kind(r.get("origin")) not in (None, "human"):
                pass  # sent by a plugin, a background task or another agent, not typed by the person
            elif has_result and not first.startswith(FORK):
                pass
            elif first.startswith(COMMAND):
                close_turn()
                turn = _Turn()
                awaiting_reply = True
            else:
                close_turn()
                turn = _Turn()
                seg.counts["prompts"] += 1

        elif kind == "system" and r.get("subtype") == "compact_boundary":
            seg.counts["compactions"] += 1

        elif kind == "attachment":
            att = r.get("attachment") or {}
            context(att)
            if (att.get("type") == "queued_command" and att.get("commandMode") == "prompt"
                    and _kind(att.get("origin")) == "human"):  # typed while Claude was still working
                seg.counts["prompts"] += 1

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                r = json.loads(line)
                if isinstance(r, dict):
                    handle(r)
            except BAD_RECORD:
                continue

    close_turn()
    done.append(seg.result(dict(base)))
    return [s for s in done if s]
