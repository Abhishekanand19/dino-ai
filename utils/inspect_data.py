"""
UTILITY — Inspect collected data
=================================
Run this anytime to see stats about your collected data
and preview some sample frames.

  python utils/inspect_data.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import cv2
import numpy as np

LABELS_FILE = "data/labels.csv"


def main():
    if not os.path.exists(LABELS_FILE):
        print(f"❌  No data found at {LABELS_FILE}")
        print("    Run data_collection/collect.py first!")
        return

    df = pd.read_csv(LABELS_FILE)
    total = len(df)
    valid = df[df["frame_path"].apply(os.path.exists)]

    print("=" * 50)
    print("  📊  Data Inspection Report")
    print("=" * 50)
    print(f"\n  Total rows in CSV : {total}")
    print(f"  Valid frame files : {len(valid)}")
    print(f"  Missing files     : {total - len(valid)}")

    print("\n  Action distribution:")
    labels = {0: "nothing", 1: "jump", 2: "duck"}
    counts = df["action"].value_counts().sort_index()
    for action, count in counts.items():
        pct = 100 * count / total
        bar = "█" * int(pct / 2)
        print(f"    {labels[action]:>8}: {count:>6}  ({pct:5.1f}%)  {bar}")

    # Estimate how many sessions
    frame_paths = df["frame_path"].tolist()
    sessions = set()
    for p in frame_paths:
        basename = os.path.basename(p)
        session_id = "_".join(basename.split("_")[:2])
        sessions.add(session_id)
    print(f"\n  Estimated sessions: {len(sessions)}")

    # Recommendation
    print("\n  Recommendation:")
    if total < 1000:
        print("  ⚠️  Collect more data! Aim for 3000+ frames (5-6 sessions).")
    elif total < 3000:
        print("  👍  Decent amount. More data will improve performance.")
    else:
        print("  ✅  Good amount of data. You're ready to train!")

    # Show sample frames
    print("\n  Showing 9 sample frames (press any key to close)...")
    sample = valid.sample(min(9, len(valid)), random_state=42)
    imgs = []
    for _, row in sample.iterrows():
        img = cv2.imread(row["frame_path"], cv2.IMREAD_GRAYSCALE)
        if img is not None:
            action_text = labels[int(row["action"])]
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # Color-code by action
            colors = {0: (100,100,100), 1: (0,200,0), 2: (0,100,255)}
            cv2.putText(img_bgr, action_text, (2, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        colors[int(row["action"])], 1)
            imgs.append(img_bgr)

    if imgs:
        # Pad to 9 if needed
        while len(imgs) < 9:
            imgs.append(np.zeros((80, 80, 3), dtype=np.uint8))
        grid_rows = [np.hstack(imgs[i:i+3]) for i in range(0, 9, 3)]
        grid = np.vstack(grid_rows)
        grid = cv2.resize(grid, (480, 480), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Sample Frames (gray=nothing, green=jump, orange=duck)", grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
