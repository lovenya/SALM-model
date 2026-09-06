#!/usr/bin/env python
"""
Export a slice of a Claude Code session log as readable Markdown, for a human reviewer.

WHY A SCRIPT. The first three parts of this project's transcript were exported with an inline
heredoc that was not kept, so each new part meant rewriting the same JSONL walk and rediscovering
the same shape quirks (content is sometimes a string and sometimes a list of typed blocks; tool
results and sidechains have to be dropped or the output is unreadable). This is that walk, saved.

Tool CALLS are kept but collapsed to a one-line summary inside a <details> block: a reviewer
needs to see that a claim was checked against something, not to read 500 lines of log output.
Tool RESULTS are dropped entirely -- they are the bulk of the file and almost none of it is
readable prose. Where a number matters it is in the assistant's own text.

Usage:
  python scripts/session/export_transcript.py --session <file.jsonl> --start-line N --out X.md \
      [--title "..."] [--intro "..."]

  --start-line is EXCLUSIVE: pass the line of the message that requested the previous export,
  and this picks up after it. Find it with:  grep -n "<some phrase>" <file.jsonl>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def render_user(prose: str) -> str | None:
    """
    Turn one raw user record into reviewer-facing Markdown, or None to drop it.

    Three kinds of record arrive typed as "user" and only one is the human talking:
      - the compaction summary the harness injects when context runs out. Thousands of words
        of internal notes-to-self; including it makes the transcript unreadable and leaks
        working state a reviewer should not have to wade through.
      - shell commands the user ran inline with `!`, wrapped in <bash-input> tags. These ARE
        the user acting and belong in the record, but as a command block, not raw tags.
      - actual typed prose.
    """
    t = prose.strip()
    if t.startswith("This session is being continued from a previous conversation"):
        return None
    if t.startswith("Base directory for this skill:"):
        return None          # skill body injected by the harness, not typed by the human
    if any(tag in t for tag in ("<bash-input>", "<bash-stdout>", "<bash-stderr>")):
        cmds = re.findall(r"<bash-input>(.*?)</bash-input>", t, re.S)
        outs = re.findall(r"<bash-stdout>(.*?)</bash-stdout>", t, re.S)
        errs = re.findall(r"<bash-stderr>(.*?)</bash-stderr>", t, re.S)
        body = [c.strip() for c in cmds if c.strip()]
        for o in outs + errs:
            # Slurm colourises its errors; the escape codes are noise in a Markdown file.
            o = re.sub(r"(?:\x1b)?\[[0-9;]*m", "", o).strip()
            body += [f"# -> {ln.strip()}" for ln in o.splitlines() if ln.strip()]
        if not body:
            return None
        label = "*(command run by the user)*" if cmds else "*(output of the command above)*"
        return "\n".join([label, "", "```bash", *body, "```"])
    return t or None


def block_text(content) -> tuple[str, list[str]]:
    """Return (prose, tool_call_descriptions) for one message's content field."""
    if isinstance(content, str):
        return content, []
    prose: list[str] = []
    calls: list[str] = []
    for b in content if isinstance(content, list) else []:
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t == "text":
            prose.append(b.get("text", ""))
        elif t == "thinking":
            pass  # deliberately omitted: reasoning drafts are not the record
        elif t == "tool_use":
            name = b.get("name", "?")
            desc = (b.get("input") or {}).get("description") or ""
            calls.append(f"`{name}`" + (f" — {desc}" if desc else ""))
    return "\n".join(p for p in prose if p.strip()), calls


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--start-line", type=int, required=True,
                    help="exclusive; export begins after this 1-indexed line")
    ap.add_argument("--end-line", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Session Transcript")
    ap.add_argument("--intro", default="")
    args = ap.parse_args()

    lines = Path(args.session).read_text(errors="replace").splitlines()
    end = args.end_line or len(lines)
    out: list[str] = [f"# {args.title}", ""]
    if args.intro:
        out += [args.intro, ""]
    out += ["---", ""]

    n_user = n_asst = n_calls = 0
    for raw in lines[args.start_line:end]:
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if rec.get("isSidechain"):
            continue                      # subagent chatter is not part of the conversation

        att = rec.get("attachment") or {}
        if att.get("type") == "queued_command" and (att.get("origin") or {}).get("kind") == "human":
            # Sent mid-turn, while the assistant was still working. Stored outside the normal
            # message stream, and frequently the turn that redirects everything -- dropping
            # these silently was the exporter's worst bug.
            txt = (att.get("prompt") or "").strip()
            if txt:
                n_user += 1
                out += ["## 🧑 User *(sent mid-turn)*", "", txt, ""]
            continue

        if rec.get("type") not in ("user", "assistant"):
            continue
        msg = rec.get("message") or {}
        prose, calls = block_text(msg.get("content"))

        if rec["type"] == "user":
            # A "user" record carrying only tool_result blocks is the harness replying to a
            # tool call, not the human. Those are the majority of user-typed records.
            if not prose.strip():
                continue
            if prose.lstrip().startswith(("<system-reminder>", "<task-notification>",
                                          "[SYSTEM NOTIFICATION")):
                continue
            rendered = render_user(prose)
            if rendered is None:
                continue
            n_user += 1
            out += ["## 🧑 User", "", rendered, ""]
        else:
            if prose.strip():
                n_asst += 1
                out += ["### 🤖 Claude", "", prose.strip(), ""]
            if calls:
                n_calls += len(calls)
                out += ["<details><summary>tool calls</summary>", ""]
                out += [f"- {c}" for c in calls]
                out += ["", "</details>", ""]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(out))
    kb = Path(args.out).stat().st_size / 1024
    print(f"wrote {args.out}  ({kb:.0f} KB)")
    print(f"  {n_user} user messages, {n_asst} assistant messages, {n_calls} tool calls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
