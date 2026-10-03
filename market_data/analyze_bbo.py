import json
from pathlib import Path
from collections import Counter

FILE = Path("data/raw/btc_88000_5min.jsonl")
TOKEN_ID = "41204045870451989441006259077692571105773543114590728783061562183667482414716"

bbo_rows = []
event_counts = Counter()
invalid = 0

with FILE.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        record = json.loads(line)
        event = record.get("event", {})
        event_type = event.get("event_type", "unknown")
        event_counts[event_type] += 1

        # Initial order-book snapshot
        if event_type == "book" and event.get("asset_id") == TOKEN_ID:
            bids = event.get("bids", [])
            asks = event.get("asks", [])
            if bids and asks:
                bid = max(float(x["price"]) for x in bids)
                ask = min(float(x["price"]) for x in asks)
                bbo_rows.append((int(event["timestamp"]), bid, ask, "book"))

        # Incremental updates: only selected token
        elif event_type == "price_change":
            for change in event.get("price_changes", []):
                if change.get("asset_id") != TOKEN_ID:
                    continue

                bid = change.get("best_bid")
                ask = change.get("best_ask")
                if bid is None or ask is None:
                    invalid += 1
                    continue

                bid, ask = float(bid), float(ask)
                bbo_rows.append(
                    (int(event["timestamp"]), bid, ask, "price_change")
                )

bbo_rows.sort(key=lambda x: x[0])

print("Event counts:", dict(event_counts))
print("Selected-token BBO observations:", len(bbo_rows))
print("Missing BBO fields:", invalid)

if not bbo_rows:
    print("No BBO observations found.")
    raise SystemExit

pairs = [(x[1], x[2]) for x in bbo_rows]
mids = [(bid + ask) / 2 for bid, ask in pairs]
spreads = [ask - bid for bid, ask in pairs]

print("Unique BBO pairs:", len(set(pairs)))
print("Unique mid prices:", len(set(mids)))
print(f"First BBO: bid={bbo_rows[0][1]}, ask={bbo_rows[0][2]}")
print(f"Last BBO:  bid={bbo_rows[-1][1]}, ask={bbo_rows[-1][2]}")
print(f"Bid range: {min(x[1] for x in bbo_rows):.4f} to {max(x[1] for x in bbo_rows):.4f}")
print(f"Ask range: {min(x[2] for x in bbo_rows):.4f} to {max(x[2] for x in bbo_rows):.4f}")
print(f"Mid range: {min(mids):.4f} to {max(mids):.4f}")
print(f"Average spread: {sum(spreads) / len(spreads):.6f}")
print("Crossed BBO observations:", sum(1 for bid, ask in pairs if bid >= ask))

changes = sum(1 for i in range(1, len(pairs)) if pairs[i] != pairs[i - 1])
print("Consecutive BBO changes:", changes)

print("\nFirst 5 observations:")
for row in bbo_rows[:5]:
    print(row)

print("\nLast 5 observations:")
for row in bbo_rows[-5:]:
    print(row)