import json
from collections import Counter
from pathlib import Path

FILE = Path("data/raw/btc_88000_5min.jsonl")

event_types = Counter()
received_times = []
event_keys = Counter()
samples = {}

with FILE.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        record = json.loads(line)
        received_at = record.get("received_at")
        if received_at:
            received_times.append(received_at)

        event = record.get("event", {})
        if isinstance(event, dict):
            event_type = event.get("event_type", event.get("type", "unknown"))
            event_types[event_type] += 1

            for key in event:
                event_keys[key] += 1

            samples.setdefault(event_type, event)

print(f"File: {FILE}")
print(f"Total records: {sum(event_types.values())}")

if received_times:
    print(f"First received: {min(received_times)}")
    print(f"Last received:  {max(received_times)}")

print("\nEvent types:")
for event_type, count in event_types.most_common():
    print(f"  {event_type}: {count}")

print("\nCommon event fields:")
for key, count in event_keys.most_common():
    print(f"  {key}: {count}")

print("\nSample event from each type:")
for event_type, event in samples.items():
    print(f"\n--- {event_type} ---")
    print(json.dumps(event, indent=2)[:2500])