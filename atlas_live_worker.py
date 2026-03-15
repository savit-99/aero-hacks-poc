import argparse
import time

from db import atlas_enabled, run_change_stream_once


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events_per_cycle", type=int, default=20)
    parser.add_argument("--cycle_seconds", type=int, default=8)
    parser.add_argument("--cycles", type=int, default=0, help="0 means run forever")
    args = parser.parse_args()

    if not atlas_enabled():
        print("Atlas disabled: MONGODB_URI not found.")
        return

    cycle = 0
    while True:
        cycle += 1
        consumed = run_change_stream_once(
            max_events=max(1, int(args.events_per_cycle)),
            max_seconds=max(1, int(args.cycle_seconds)),
        )
        print(f"[cycle={cycle}] consumed_events={consumed}")
        if args.cycles > 0 and cycle >= int(args.cycles):
            break
        time.sleep(0.8)


if __name__ == "__main__":
    main()
