"""
STEP 3 — Inference (The Magic Moment 🦕)
=========================================
This is where the AI plays the game with NO keyboard, NO mouse from you.

HOW TO USE:
  1. Open Chrome → chrome://dino
  2. Make sure CAPTURE_REGION in this file matches your setup
     (same as what you set in collect.py)
  3. Run: python inference/play.py
  4. Press SPACE in the terminal to start the AI
  5. Watch it play — start your screen recording!
  6. Press ESC to stop

The AI will:
  - Capture the screen at ~15 FPS
  - Feed each frame through DinoNet
  - Press SPACE (jump) or DOWN ARROW (duck) automatically
  - You touch nothing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import cv2
import mss
import pyautogui
import keyboard
import time

from utils.model import DinoNet, get_device

# ─── CONFIG ───────────────────────────────────────────────────────────────────

# Must match what you used in collect.py
CAPTURE_REGION = {"top": 300, "left": 0, "width": 800, "height": 300}

MODEL_PATH = "models/dino_model.pth"

# Inference FPS — how fast the AI reacts
FPS = 15

# Minimum confidence to act (0.0–1.0). Raise if AI acts too jittery.
CONFIDENCE_THRESHOLD = 0.6

# How long to hold a key press (seconds)
JUMP_DURATION = 0.1
DUCK_DURATION = 0.1

# ──────────────────────────────────────────────────────────────────────────────

ACTION_NAMES = {0: "nothing", 1: "JUMP  🦘", 2: "DUCK  🦆"}


def preprocess_frame(frame):
    """Same preprocessing as during data collection — must match exactly."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (80, 80))
    tensor = resized.astype(np.float32) / 255.0
    tensor = np.expand_dims(tensor, axis=0)   # (1, 80, 80)
    tensor = np.expand_dims(tensor, axis=0)   # (1, 1, 80, 80)
    return torch.tensor(tensor, dtype=torch.float32)


def press_action(action):
    """Execute the predicted action as a real keypress."""
    if action == 1:
        pyautogui.keyDown("space")
        time.sleep(JUMP_DURATION)
        pyautogui.keyUp("space")
    elif action == 2:
        pyautogui.keyDown("down")
        time.sleep(DUCK_DURATION)
        pyautogui.keyUp("down")
    # action == 0: do nothing


def load_model(device):
    """Load trained DinoNet weights."""
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌  Model not found at {MODEL_PATH}")
        print("    Run training/train.py first!\n")
        sys.exit(1)

    model = DinoNet(num_classes=3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print(f"✅  Model loaded from {MODEL_PATH}")
    return model


def run():
    print("=" * 55)
    print("  🦕  Chrome Dino AI — Autonomous Play")
    print("=" * 55)

    device = get_device()
    model  = load_model(device)

    print(f"\n⚙️   FPS: {FPS}  |  Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print("\nSteps:")
    print("  1. Open Chrome → chrome://dino")
    print("  2. Keep Chrome visible on screen")
    print("  3. Press SPACE here to unleash the AI")
    print("  4. START your screen recording!")
    print("  5. Press ESC to stop\n")

    print("Waiting for SPACE to begin...")
    keyboard.wait("space")

    # Give 3 seconds so you can click Chrome / start screen recorder
    print("\nStarting in 3...")
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1...")
    time.sleep(1)
    print("\n🚀  AI is playing! Press ESC to stop.\n")

    # Start the dino game (first space press starts it)
    pyautogui.press("space")
    time.sleep(0.5)

    sct       = mss.mss()
    interval  = 1.0 / FPS
    frame_num = 0
    action_counts = {0: 0, 1: 0, 2: 0}

    try:
        while not keyboard.is_pressed("esc"):
            loop_start = time.time()

            # Capture
            raw   = sct.grab(CAPTURE_REGION)
            frame = np.array(raw)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Predict
            with torch.no_grad():
                input_tensor = preprocess_frame(frame).to(device)
                logits       = model(input_tensor)
                probs        = torch.softmax(logits, dim=1)[0]
                confidence, predicted = torch.max(probs, dim=0)
                action       = predicted.item()
                conf_val     = confidence.item()

            # Only act if confident enough
            if conf_val < CONFIDENCE_THRESHOLD:
                action = 0

            # Execute
            press_action(action)
            action_counts[action] += 1

            # Live status every 30 frames
            if frame_num % 30 == 0:
                total = sum(action_counts.values())
                print(f"Frame {frame_num:>5}  |  Action: {ACTION_NAMES[action]:<12}  |  "
                      f"Conf: {conf_val:.2f}  |  "
                      f"J:{action_counts[1]}  D:{action_counts[2]}  N:{action_counts[0]}")

            frame_num += 1

            # Maintain FPS
            elapsed = time.time() - loop_start
            wait    = interval - elapsed
            if wait > 0:
                time.sleep(wait)

    except KeyboardInterrupt:
        pass
    finally:
        # Make sure no keys are stuck down
        pyautogui.keyUp("space")
        pyautogui.keyUp("down")

        total = sum(action_counts.values())
        print(f"\n\n📊  Session summary ({total} frames):")
        for a, name in ACTION_NAMES.items():
            pct = 100 * action_counts[a] / total if total > 0 else 0
            print(f"     {name:<15}: {action_counts[a]:>5} ({pct:.1f}%)")
        print("\n👋  AI stopped. Nice run!\n")


if __name__ == "__main__":
    run()
