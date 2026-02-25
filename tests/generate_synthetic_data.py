"""Generate a synthetic process mining dataset for end-to-end testing.

Simulates business process cases with events:
  NEW -> ACCEPTED -> BILLING -> DELIVERY -> DELIVERED  (happy path)
  NEW -> ACCEPTED -> BILLING -> CANCELED               (cancel path)
  NEW -> REJECTED                                       (reject path)

Output: CSV with __EVENT_1 .. __EVENT_N and __DURATION_EVENT_1 .. __DURATION_EVENT_N columns.
"""
import random
import csv
import os

EVENTS_HAPPY = ["New", "Accepted", "Billing", "Delivery", "Delivered"]
EVENTS_CANCEL = ["New", "Accepted", "Billing", "Canceled"]
EVENTS_REJECT = ["New", "Rejected"]

PATHS = [
    (EVENTS_HAPPY, 0.55),
    (EVENTS_CANCEL, 0.30),
    (EVENTS_REJECT, 0.15),
]

MAX_EVENTS = max(len(p[0]) for p in PATHS)
NUM_CASES = 500
SEED = 42

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "synthetic_process_data.csv")


def generate_duration(base_hours, jitter_hours):
    return max(0.1, base_hours + random.uniform(-jitter_hours, jitter_hours))


def generate_case():
    r = random.random()
    cumulative = 0.0
    chosen_path = EVENTS_HAPPY
    for path, prob in PATHS:
        cumulative += prob
        if r <= cumulative:
            chosen_path = path
            break

    events = list(chosen_path)
    durations = []
    cumulative_hours = 0.0
    for i in range(len(events)):
        step_duration = generate_duration(base_hours=2.0 * (i + 1), jitter_hours=1.5)
        cumulative_hours += step_duration
        durations.append(round(cumulative_hours, 2))

    return events, durations


def main():
    random.seed(SEED)

    event_cols = [f"__EVENT_{i+1}" for i in range(MAX_EVENTS)]
    duration_cols = [f"__DURATION_EVENT_{i+1}" for i in range(MAX_EVENTS)]
    header = event_cols + duration_cols

    rows = []
    for _ in range(NUM_CASES):
        events, durations = generate_case()
        event_row = events + [""] * (MAX_EVENTS - len(events))
        duration_row = durations + [0] * (MAX_EVENTS - len(durations))
        rows.append(event_row + duration_row)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Generated {NUM_CASES} cases -> {OUTPUT_CSV}")
    print(f"Columns: {header}")

    # Print distribution
    path_counts = {}
    for row in rows:
        events = [e for e in row[:MAX_EVENTS] if e]
        key = " -> ".join(events)
        path_counts[key] = path_counts.get(key, 0) + 1
    print("\nPath distribution:")
    for path, count in sorted(path_counts.items(), key=lambda x: -x[1]):
        print(f"  {path}: {count} ({count/NUM_CASES*100:.1f}%)")


if __name__ == "__main__":
    main()
