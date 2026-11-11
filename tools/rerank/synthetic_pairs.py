"""Synthetic rerank requests for eklow's Qwen3-Reranker checks (written for these tests; no real data in them).

Usage: python3 tools/rerank/synthetic_pairs.py OUT.json

A list of {"name", "query", "documents"} in the shape POST /v1/rerank takes. Each query has a relevant document, near
misses (the same topic, not the answer), an indirect match (the answer must be inferred), unrelated documents, and a
non-English or emoji document; two latency requests carry 8 and 24 documents of about 1,000 characters.
"""

import json
import sys

FILLER = ("The household notes are kept in a shared folder and reviewed on Sundays. Each entry records who changed it "
          "and why, so that a reminder that fires at the wrong time can be traced back to the change that caused it. ")


def para(seed, n=1000):
    words = ["garden", "boiler", "invoice", "train", "recipe", "battery", "window", "ticket", "meeting", "router",
             "kettle", "parcel", "doctor", "school", "bicycle", "laptop", "insurance", "holiday", "budget", "printer"]
    out, i = [], seed
    while sum(len(w) + 1 for w in out) < n:
        i = (i * 1103515245 + 12345) % 2147483648
        out.append(words[i % len(words)])
        if i % 7 == 0:
            out[-1] += "."
    s = " ".join(out)
    return s[:n]


REQUESTS = [
    {"name": "boiler", "query": "When was the boiler last serviced?",
     "documents": [
         "Boiler service log: the annual service was done on 14 March 2026 by Hargreaves Heating; next due March 2027.",
         "The boiler makes a clicking noise when the heating first comes on in the morning.",
         "Gas bill for February: 182 kWh, paid by direct debit on the 3rd.",
         "The engineer who came in spring replaced the pressure valve and said everything else was fine.",
         "Recipe: lemon drizzle cake with 225 g of self-raising flour and two lemons.",
         "Train times from Leeds to York on Saturdays: every 20 minutes from 07:10.",
         "Wartung der Heizung: Termin am 14. März 2026 erledigt, nächste Wartung im März 2027.",
         "🔥🛠️ boiler ok ✅"]},
    {"name": "dentist", "query": "What time is Valerie's dentist appointment on Thursday?",
     "documents": [
         "Calendar: Thursday 12 Nov, 15:30, Dr. Muñoz (dentist) for Valerie; bring the insurance card.",
         "Valerie prefers afternoon appointments because of her morning classes.",
         "Dentist's reminder: please arrive ten minutes early for your check-up.",
         "Thursday: bins go out, recycling is collected on alternate weeks.",
         "The dentist on Calle Mayor charges 20 euros to reschedule.",
         "How to descale a kettle with citric acid.",
         "歯医者の予約は木曜日の午後3時半です。"]},
    {"name": "router", "query": "how do I reset the wifi router",
     "documents": [
         "To reset the router, hold the recessed button on the back for 10 seconds until the lights flash amber, then wait two minutes.",
         "The router is in the hallway cupboard next to the electricity meter.",
         "Our broadband contract renews in June; the provider can be called on 0800 123 456.",
         "If the wifi drops in the evening, the neighbours' network on channel 6 may be interfering.",
         "Pressing the WPS button pairs a new device without a password.",
         "Bicycle maintenance: oil the chain every 300 km.",
         "A list of birthday presents for the twins."]},
    {"name": "backup", "query": "Did the nightly backup job succeed last night?",
     "documents": [
         '[{"id": "nightly-backup", "last_run": "2026-03-04T02:00:11Z", "status": "ok", "duration_s": 412}]',
         '[{"id": "nightly-backup", "last_run": "2026-03-03T02:00:09Z", "status": "failed", "error": "disk full"}]',
         "Backups are kept for 30 days on the NAS and for a year in cold storage.",
         "The weekly report job runs on Mondays at 09:00.",
         "Photos from the beach holiday, sorted by day.",
         "A haiku: autumn moonlight / a worm digs silently / into the chestnut."]},
    {"name": "latency-8", "query": "Which document explains how to renew the home insurance policy?",
     "documents": [FILLER + "To renew the home insurance, log in to the insurer's site before the 1st of the month, "
                   "confirm the contents value and pay the premium; the documents arrive by post. " + para(1, 700)]
     + [FILLER + para(s) for s in range(2, 9)]},
    {"name": "latency-24", "query": "Where is the spare key for the shed kept?",
     "documents": [FILLER + "The spare key for the shed hangs on the hook inside the pantry door, labelled in red. "
                   + para(31, 750)] + [FILLER + para(s) for s in range(32, 55)]},
]

if __name__ == "__main__":
    json.dump(REQUESTS, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
