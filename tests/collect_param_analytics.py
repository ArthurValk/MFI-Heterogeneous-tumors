#!/usr/bin/env python3
"""Collect analytics on parameter combination runtimes."""

import csv
from pathlib import Path
from datetime import datetime

param_sweep_dir = Path("./tests/test_output/param_sweep")

# Get all subdirectories with their modification times
combos = []
for combo_dir in sorted(param_sweep_dir.iterdir()):
    if combo_dir.is_dir():
        mtime = combo_dir.stat().st_mtime
        mtime_dt = datetime.fromtimestamp(mtime)
        combos.append({
            "name": combo_dir.name,
            "timestamp": mtime,
            "datetime": mtime_dt.isoformat(),
        })

# Sort by timestamp to get execution order
combos.sort(key=lambda x: x["timestamp"])

# Calculate duration for each (time until next combo started)
for i in range(len(combos) - 1):
    duration = combos[i + 1]["timestamp"] - combos[i]["timestamp"]
    combos[i]["duration_seconds"] = duration
    combos[i]["duration_minutes"] = round(duration / 60, 2)

# Last combo: use current time as reference
if combos:
    last_duration = datetime.now().timestamp() - combos[-1]["timestamp"]
    combos[-1]["duration_seconds"] = last_duration
    combos[-1]["duration_minutes"] = round(last_duration / 60, 2)

# Sort by duration to find slowest
slowest = sorted(combos, key=lambda x: x.get("duration_seconds", 0), reverse=True)

# Write to CSV
with open("param_analytics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["rank", "execution_order", "parameter_combination", "duration_minutes", "duration_seconds", "completed_at"])

    for rank, combo in enumerate(slowest, 1):
        # Find execution order
        exec_order = next(i + 1 for i, c in enumerate(combos) if c["name"] == combo["name"])
        writer.writerow([
            rank,
            exec_order,
            combo["name"],
            combo.get("duration_minutes", ""),
            round(combo.get("duration_seconds", 0), 1),
            combo["datetime"],
        ])

# Write summary to txt
with open("param_analytics.txt", "w") as f:
    f.write("=" * 100 + "\n")
    f.write("PARAMETER SWEEP ANALYTICS SUMMARY\n")
    f.write("=" * 100 + "\n\n")

    # Statistics
    durations = [combo.get("duration_minutes", 0) for combo in combos]
    avg_duration = sum(durations) / len(durations) if durations else 0
    max_duration = max(durations) if durations else 0
    min_duration = min(durations) if durations else 0

    f.write(f"Total combinations completed: {len(combos)}\n")
    f.write(f"Average time per combination: {avg_duration:.2f} minutes\n")
    f.write(f"Slowest combination: {max_duration:.2f} minutes\n")
    f.write(f"Fastest combination: {min_duration:.2f} minutes\n")
    f.write(f"Speed variation: {max_duration / min_duration:.2f}x\n\n")

    f.write("=" * 100 + "\n")
    f.write("TOP 10 SLOWEST PARAMETER COMBINATIONS:\n")
    f.write("-" * 100 + "\n")
    for i, combo in enumerate(slowest[:10], 1):
        f.write(f"{i}. {combo['name']}\n")
        f.write(f"   Duration: {combo.get('duration_minutes', 'N/A')} minutes\n")
        f.write(f"   Completed: {combo['datetime']}\n\n")

    f.write("\n" + "=" * 100 + "\n")
    f.write("TOP 10 FASTEST PARAMETER COMBINATIONS:\n")
    f.write("-" * 100 + "\n")
    fastest = sorted(combos, key=lambda x: x.get("duration_seconds", 0))
    for i, combo in enumerate(fastest[:10], 1):
        f.write(f"{i}. {combo['name']}\n")
        f.write(f"   Duration: {combo.get('duration_minutes', 'N/A')} minutes\n")
        f.write(f"   Completed: {combo['datetime']}\n\n")

    f.write("\nDetailed results saved to: param_analytics.csv\n")

print("Analytics saved!")
print(f"  - Detailed CSV: param_analytics.csv")
print(f"  - Summary TXT: param_analytics.txt")
print(f"\nStatistics:")
print(f"  Total combinations: {len(combos)}")
print(f"  Average time per combination: {avg_duration:.2f} minutes")
print(f"  Slowest: {max_duration:.2f} minutes")
print(f"  Fastest: {min_duration:.2f} minutes")
print(f"  Speed variation: {max_duration / min_duration:.2f}x")
print(f"\nTop 5 slowest combinations:")
for i, combo in enumerate(slowest[:5], 1):
    print(f"{i}. {combo['duration_minutes']} min - {combo['name']}")
