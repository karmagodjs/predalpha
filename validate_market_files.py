from pathlib import Path
import json

files = sorted(Path("data/raw").glob("btc-updown-5m-*.jsonl"))
print("Total market files:", len(files))

for path in files:
    count = 0
    bad = 0
    first = last = None

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                count += 1
                ts = obj.get("timestamp")
                if ts is not None:
                    first = first or ts
                    last = ts
            except Exception:
                bad += 1

    print(f"{path.name} | records={count} | bad={bad} | first={first} | last={last}")
