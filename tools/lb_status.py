"""Print the latest Kaggle submission as `lb:<ref>:<status>:<public score>` (for progress_log.sh).

kaggle competitions submissions -c gemma-4-developer-agent --csv | python3 tools/lb_status.py
"""

import csv
import sys

rows = list(csv.reader(sys.stdin))
# the CLI may print notices before the CSV header: start after the header row
start = next((i for i, r in enumerate(rows) if r and r[0] in ("fileName", "ref")), -1) + 1
data = [r for r in rows[start:] if len(r) > 5] if start else []
if not data:
    print("lb:?")
else:
    r = data[0]
    print(f"lb:{r[0]}:{r[4].replace('SubmissionStatus.', '')}:{r[5] or '-'}")
