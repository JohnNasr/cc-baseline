"""Local SQLite store: one row per transcript file, kept even after Claude Code deletes the file."""
import glob
import json
import os
import sqlite3
import time

from .parse import PARSER_VERSION, parse_file


def default_db():
    return os.environ.get("CC_BASELINE_DB") or os.path.join(os.path.expanduser("~"), ".cc-baseline", "baseline.db")


def default_root():
    config = os.environ.get("CLAUDE_CONFIG_DIR") or os.path.join(os.path.expanduser("~"), ".claude")
    return os.path.join(config, "projects")


def open_db(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    con = sqlite3.connect(path, timeout=10)
    con.execute("CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, mtime REAL, size INTEGER, parser INTEGER, data TEXT)")
    con.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)")
    return con


BIG_FILE = 100 * 1024 * 1024  # the hook leaves bigger transcripts to `cc-baseline ingest`
FLUSH_S = 2  # commit at least this often, so a run cut short keeps its work and readers are not locked out


def ingest(con, root, deadline=None, skip_big=False):
    """Parse new or changed transcripts. Returns (parsed, skipped_unchanged, failed, unfinished)."""
    known = {p: (m, s, v) for p, m, s, v in con.execute("SELECT path, mtime, size, parser FROM files")}
    parsed = skipped = failed = 0
    unfinished = False
    rows, flushed = [], time.monotonic()
    for found in glob.glob(os.path.join(glob.escape(root), "**", "*.jsonl"), recursive=True):
        path = os.path.normcase(os.path.realpath(found))  # one key per file however --root is spelled
        try:
            st = os.stat(path)
        except OSError:
            continue
        if known.get(path) == (st.st_mtime, st.st_size, PARSER_VERSION):
            skipped += 1
            continue
        if skip_big and st.st_size > BIG_FILE:
            continue
        if deadline is not None and time.monotonic() > deadline:
            unfinished = True
            break
        try:
            data = json.dumps(parse_file(path))
            parsed += 1
        except OSError:  # locked or vanished: try again next run
            failed += 1
            continue
        except Exception:  # a file the parser cannot handle is not retried until it changes
            failed += 1
            data = None
        rows.append((path, st.st_mtime, st.st_size, PARSER_VERSION, data))
        if time.monotonic() - flushed > FLUSH_S:
            _write(con, rows)
            rows, flushed = [], time.monotonic()
    _write(con, rows)
    return parsed, skipped, failed, unfinished


def _write(con, rows):
    with con:
        con.executemany("INSERT OR REPLACE INTO files VALUES (?, ?, ?, ?, ?)", rows)


def load_sessions(con):
    """Sessions from the current parser only: older rows counted some metrics differently."""
    rows = con.execute("SELECT data FROM files WHERE data IS NOT NULL AND parser = ?", (PARSER_VERSION,))
    return [seg for (d,) in rows for seg in json.loads(d) or []]


def stale_files(con):
    """Rows from an older parser whose transcript is gone, so they cannot be re-parsed."""
    return con.execute("SELECT COUNT(*) FROM files WHERE parser != ?", (PARSER_VERSION,)).fetchone()[0]


def get_meta(con, key):
    row = con.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def set_meta(con, key, value):
    con.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))
    con.commit()
