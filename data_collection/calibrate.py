"""
CALIBRATION TOOL
================
Run this to figure out the correct CAPTURE_REGION for your screen.
It takes a screenshot and draws a rectangle so you can adjust coordinates.

HOW TO USE:
  python data_collection/calibrate.py
  → A window will open showing your screen with a red box
  → Adjust TOP, LEFT, WIDTH, HEIGHT in collect.py to frame the game
"""

import mss
import cv2
import numpy as np

# ── Tweak these until the red box perfectly frames the dino game ──
TOP    = 300
LEFT   = 0
WIDTH  = 800
HEIGHT = 300
# ─────────────────────────────────────────────────────────────────

def main():
    print("Taking screenshot to help you calibrate the capture region...")
    print(f"Current region: top={TOP}, left={LEFT}, width={WIDTH}, height={HEIGHT}")
    print("\nA window will open. Adjust TOP/LEFT/WIDTH/HEIGHT in this file,")
    print("then copy the values into data_collection/collect.py")
    print("\nPress any key in the image window to close it.\n")

    with mss.mss() as sct:
        # Grab full screen first
        monitor = sct.monitors[1]  # primary monitor
        full = np.array(sct.grab(monitor))
        full = cv2.cvtColor(full, cv2.COLOR_BGRA2BGR)

        # Draw the capture region in red
        x1, y1 = LEFT, TOP
        x2, y2 = LEFT + WIDTH, TOP + HEIGHT
        cv2.rectangle(full, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(full, "Capture Region", (x1 + 5, y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Scale down for display if monitor is large
        h, w = full.shape[:2]
        scale = min(1.0, 1200 / w)
        display = cv2.resize(full, (int(w * scale), int(h * scale)))

        cv2.imshow("Calibration — Adjust region so red box frames the game", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"\nOnce happy, set in collect.py:")
    print(f'  CAPTURE_REGION = {{"top": {TOP}, "left": {LEFT}, "width": {WIDTH}, "height": {HEIGHT}}}')


if __name__ == "__main__":
    main()
