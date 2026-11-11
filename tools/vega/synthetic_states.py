"""Synthetic end-of-turn checker states, written for eklow's tests (no real conversation is in them).

Usage: python3 tools/vega/synthetic_states.py OUT.json

Each state has the exact shape son-of-anton's checker sends (daemon/internal/check State(): the tools, the user's latest
message, this turn's calls and results, the draft), so the reader sees the token mix it will see in production. The set
covers the prompt builder's branches by length: a state under 128 tokens (no memory chunks), a few hundred (chunks,
the question read first), ~2.5k, and two past 4,096 tokens (no question first; the state read once as a prefix).
Non-ASCII text (accents, CJK, emoji, a combining mark) and CRLF line ends are in two of them.
"""

import json
import sys

TOOLS = ["read_file", "write_file", "patch", "search_files", "todo", "clarify", "memory", "session_search", "web_search",
         "web_extract", "cronjob", "relay"]


def cut(s, n):
    return s if len(s) <= n else s[:n] + " [...]"


def state(user, steps, draft, tools=TOOLS):
    turn = []
    for calls, results in steps:
        turn += ["call %s %s" % (name, cut(args, 600)) for name, args in calls]
        turn += ["result " + cut(r, 1500) for r in results]
    return ("Tools the assistant can call: " + ", ".join(tools) + "\n\n"
            "The user's latest message:\n" + cut(user, 3000) + "\n\n"
            "This turn's tool calls and results before the draft:\n" + ("\n".join(turn) if turn else "(none)") + "\n\n"
            "The assistant's draft reply (the turn ends with it; it makes no tool call):\n" + cut(draft, 3000))


def log_lines(n, start=7):
    out = []
    for i in range(n):
        h, m = divmod(start * 60 + i * 7, 60)
        out.append("2026-03-%02d %02d:%02d:%02d worker-%d %s queue=%d latency_ms=%d" % (
            1 + i % 28, h % 24, m, (i * 13) % 60, i % 4, ["ok", "retry", "ok", "slow", "ok"][i % 5], (i * 37) % 101, 40 + (i * 53) % 900))
    return "\n".join(out)


def recipe(n):
    steps = ["Preheat the oven to %d degrees and grease a tin of %d cm." % (160 + 10 * (i % 5), 18 + i % 6) for i in range(n)]
    return " ".join("Step %d: %s" % (i + 1, s) for i, s in enumerate(steps))


STATES = {
    "tiny": state("thanks!", [], "You're welcome."),
    "short-claim": state(
        "Can you add oat milk to the shopping list?", [],
        "Done! I've added oat milk to your shopping list, right under the coffee filters."),
    "short-checked": state(
        "Did the backup job run last night?",
        [([("cronjob", '{"action": "list"}')],
          ['[{"id": "nightly-backup", "schedule": "0 2 * * *", "last_run": "2026-03-04T02:00:11Z", "status": "ok", '
           '"duration_s": 412}, {"id": "weekly-report", "schedule": "0 9 * * 1", "last_run": "2026-03-02T09:00:02Z", '
           '"status": "ok"}]'])],
        "Yes. The nightly backup ran at 2:00 this morning and finished without errors in about seven minutes."),
    "unchecked-why": state(
        "Why did the bakery reminder go out at 6am instead of 8? It woke everyone up.", [],
        "Sorry about that! The reminder fired at 6 because the job was set in UTC rather than local time. I've "
        "moved it to 8:00 local from now on, so tomorrow's will arrive at the right time."),
    "offtopic-unicode": state(
        "¿Puedes recordarme la cita del dentista el jueves a las 15:30? 歯医者の予約も忘れずに 🦷",
        [([("memory", '{"action": "search", "query": "dentist"}')],
          ["[fact dentist-office] Dra. Muñoz, Calle Mayor 12; reschedule fee 20 €. Café nearby: «El Rincón». "
           "Noted 2026-02-11: prefers afternoon slots; é written decomposed here on purpose."])],
        "Here is the sourdough recipe you asked about last week: 500 g flour, 350 g water, 100 g starter, 10 g salt. "
        "Fold every 30 minutes for two hours, then shape and proof overnight in the fridge. 🍞"),
    "medium-logs": state(
        "The worker pool seems slow today. Is anything stuck?\r\nI saw retries in the dashboard.",
        [([("read_file", '{"path": "/var/log/pool/worker.log", "tail": 400}')], [log_lines(60)]),
         ([("search_files", '{"pattern": "retry", "path": "/var/log/pool"}')],
          ["/var/log/pool/worker.log: 12 matches\n/var/log/pool/worker.log.1: 31 matches"])],
        "Nothing is stuck. Worker-1 retried a handful of jobs this morning and worker-3 had a few slow ones over "
        "800 ms, but the queue drained each time. Yesterday's log had more retries (31) than today's (12)."),
    "long-steps": state(
        "Please go through the recipe notes and write the cleaned-up version to recipes/plum-cake.md, then tell me "
        "how long it bakes.",
        [([("read_file", '{"path": "notes/plum-cake-raw.txt"}')], [recipe(30)]),
         ([("read_file", '{"path": "notes/plum-cake-comments.txt"}')], [recipe(25)]),
         ([("search_files", '{"pattern": "bake", "path": "notes"}')], [recipe(28)]),
         ([("read_file", '{"path": "notes/oven.txt"}')], [log_lines(40, 9)])],
        "I've written the cleaned-up recipe to recipes/plum-cake.md. It bakes for 45 minutes at 180 degrees, or "
        "until a skewer comes out clean. " + recipe(6)),
    "very-long": state(
        "Summarize what changed in the worker logs this week and whether I need to do anything.",
        [([("read_file", '{"path": "/var/log/pool/worker.log.%d"}' % i)], [log_lines(45, 3 + i)]) for i in range(4)]
        + [([("session_search", '{"query": "worker pool incident"}')], [recipe(20)])],
        "This week the pool ran normally. " + log_lines(25, 11) + "\nNo action is needed."),
    "longest": state(
        "What did the logs say about worker-2 between Monday and Thursday? " + recipe(40),
        [([("read_file", '{"path": "/var/log/pool/worker.log.%d"}' % i), ("todo", '{"action": "list"}')],
          [log_lines(50, 1 + i), "- check worker-2 disk\n- rotate logs\n- reply to the landlord"]) for i in range(10)],
        "Worker-2 was quiet: " + log_lines(40, 5)),
}

if __name__ == "__main__":
    json.dump([{"key": k, "state": v} for k, v in STATES.items()], open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
