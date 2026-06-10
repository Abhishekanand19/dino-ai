"""
STEP 1 — Data Collection
========================
Run this script WHILE you play the Chrome Dino game yourself.
It records screenshots + your keypresses as training data.

HOW TO USE:
  1. Open Chrome → go to chrome://dino (or disconnect internet)
  2. Put the Chrome window on the LEFT half of your screen
  3. Run: python data_collection/collect.py
  4. Press SPACE in the terminal to START recording
  5. Play the game normally (jump with SPACE, duck with DOWN arrow)
  6. Press ESC to STOP recording
  7. Repeat 5–10 sessions to get enough data
"""

import mss
import cv2
import numpy as np
import keyboard
import time
import os
import csv
import threading
from datetime import datetime

# ─── CONFIG ───────────────────────────────────────────────────────────────────

# Screen region to capture — adjust these to match your Chrome window
# Format: {"top": Y, "left": X, "width": W, "height": H}
# TIP: Run calibrate.py first to find the right values
CAPTURE_REGION = {"top": 220, "left": 120, "width": 1600, "height": 450}

# How many frames per second to capture
FPS = 15

# Output directories
FRAMES_DIR = "data/frames"
LABELS_FILE = "data/labels.csv"

# ──────────────────────────────────────────────────────────────────────────────


def preprocess_frame(frame):
    """Convert raw screenshot to model-ready format: grayscale 80x80."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (80, 80))
    return resized


def get_action():
    """
    Returns current action as integer:
      0 = do nothing
      1 = jump (space)
      2 = duck (down arrow)
    """
    if keyboard.is_pressed("space"):
        return 1
    elif keyboard.is_pressed("down"):
        return 2
    return 0


def collect_session():
    """Runs one recording session."""
    os.makedirs(FRAMES_DIR, exist_ok=True)

    # Open CSV in append mode so multiple sessions stack up
    file_exists = os.path.exists(LABELS_FILE)
    csv_file = open(LABELS_FILE, "a", newline="")
    writer = csv.writer(csv_file)
    if not file_exists:
        writer.writerow(["frame_path", "action"])

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    frame_count = 0
    interval = 1.0 / FPS

    print(f"\n🎮  Session {session_id} started. Play the game now!")
    print("     Press ESC to stop recording.\n")

    sct = mss.mss()

    try:
        while not keyboard.is_pressed("esc"):
            start = time.time()

            # Capture screen
            raw = sct.grab(CAPTURE_REGION)
            frame = np.array(raw)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Get what key the human pressed RIGHT NOW
            action = get_action()

            # Preprocess and save frame
            processed = preprocess_frame(frame)
            filename = f"{session_id}_{frame_count:06d}.png"
            filepath = os.path.join(FRAMES_DIR, filename)
            cv2.imwrite(filepath, processed)

            # Log to CSV
            writer.writerow([filepath, action])
            frame_count += 1

            # Maintain FPS
            elapsed = time.time() - start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass
    finally:
        csv_file.close()
        print(f"\n✅  Session complete! Captured {frame_count} frames.")
        print(f"    Data saved to {LABELS_FILE}\n")


if __name__ == "__main__":
    print("=" * 55)
    print("  🦕  Chrome Dino AI — Data Collection")
    print("=" * 55)
    print("\nSteps:")
    print("  1. Open Chrome → chrome://dino")
    print("  2. Position Chrome on the LEFT side of your screen")
    print("  3. Come back to this terminal")
    print("  4. Press SPACE here to start recording")
    print("  5. Immediately click on Chrome and play!")
    print("  6. Press ESC to stop\n")

    print("Waiting for SPACE to begin...")
    keyboard.wait("space")
    time.sleep(0.5)  # tiny delay so the space press doesn't get recorded

    collect_session()

    print("Run this script again to collect more sessions.")
    print("Aim for at least 5 sessions before training.\n")
