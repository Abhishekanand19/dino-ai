# 🦕 Chrome Dino AI — Vision-Only Imitation Learning

An AI that plays Chrome Dino **with no keyboard, no mouse** — purely by looking at the screen.
Built with Python, PyTorch, and imitation learning.

---

## 📁 Project Structure

```
dino-ai/
├── data/
│   └── frames/          ← your recorded gameplay screenshots go here (auto-created)
├── data_collection/
│   ├── calibrate.py     ← FIRST: find the right screen region
│   └── collect.py       ← SECOND: record your gameplay
├── training/
│   └── train.py         ← THIRD: train the model
├── inference/
│   └── play.py          ← FOURTH: watch AI play!
├── utils/
│   ├── model.py         ← model + dataset definitions (don't edit)
│   └── inspect_data.py  ← check your data anytime
├── models/              ← trained model saved here (auto-created)
└── requirements.txt
```

---

## ⚙️ ONE-TIME SETUP

### Step 0 — Install Python & VS Code Extensions

1. Download **Python 3.10+** from https://www.python.org/downloads/
2. Open VS Code → install the **Python** extension (by Microsoft)
3. Open this folder in VS Code: `File → Open Folder → select dino-ai/`

### Step 1 — Create Virtual Environment

Open the **VS Code terminal** (`Ctrl + `` ` ```) and run:

```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# You should see (venv) at the start of your terminal line
```

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs: PyTorch, OpenCV, mss (screen capture), pyautogui, keyboard, and more.
Takes 2–5 minutes depending on your internet.

---

## 🎮 COLLECTING DATA (Days 1–2)

### Step 3 — Calibrate Screen Capture

First, find the exact screen coordinates of your Chrome Dino game window:

```bash
python data_collection/calibrate.py
```

→ A window opens showing your screen with a **red rectangle**.
→ Open the file `data_collection/calibrate.py` and adjust `TOP`, `LEFT`, `WIDTH`, `HEIGHT`
→ Re-run until the red box perfectly frames the dino game area
→ **Copy those values** into `data_collection/collect.py` → `CAPTURE_REGION`

**Tips for the best capture region:**
- Include the dino, the ground line, and the obstacle zone ahead
- Exclude browser UI (address bar, tabs) — just the game canvas
- Typical values: `top=300, left=100, width=800, height=250`

### Step 4 — Record Your Gameplay

```bash
python data_collection/collect.py
```

1. Open Chrome → go to `chrome://dino` (or turn off WiFi)
2. Run the script
3. Press **SPACE in the terminal** to start recording
4. Immediately click on Chrome and **play normally**
5. Press **ESC** to stop one session

**Do this 5–8 times.** Each session = ~200-400 frames. You want 3000+ total.

```bash
# Check how much data you have anytime:
python utils/inspect_data.py
```

**Important tips:**
- Play at your normal speed — don't go slow on purpose
- Let yourself die sometimes — dying frames are useful data too
- Try to get a good mix of jumps and ducks, not just jumps

---

## 🧠 TRAINING (Days 3–4)

### Step 5 — Train the Model

```bash
python training/train.py
```

This will:
- Load all your recorded frames
- Balance the dataset (so jump/duck aren't ignored)
- Train DinoNet for 30 epochs
- Save the best model to `models/dino_model.pth`
- Save a training plot to `models/training_plot.png`

**On CPU (integrated graphics) this takes ~5–15 minutes** depending on data size.

**What good training looks like:**
- Validation accuracy should reach 70–85%+
- Loss should decrease steadily
- If accuracy is stuck below 60%, collect more data

**Want to train faster? Use Google Colab (free GPU):**
1. Upload the `dino-ai/` folder to Google Drive
2. Open Colab → mount Drive → run `training/train.py`
3. Download `models/dino_model.pth` back to your local folder

---

## 🚀 RUNNING THE AI (Days 5–6)

### Step 6 — Watch AI Play

```bash
python inference/play.py
```

1. Open Chrome → `chrome://dino`
2. Keep Chrome **visible on screen** (don't minimize it)
3. Run the script
4. Press **SPACE in the terminal**
5. You have **3 seconds** — start your screen recorder!
6. The AI will press SPACE to start the game and play by itself
7. Press **ESC** to stop

**Tweaking AI behavior:**

If the AI jumps too late or too early, edit `inference/play.py`:
```python
CONFIDENCE_THRESHOLD = 0.6   # raise to 0.7-0.8 if too jittery
FPS = 15                     # raise to 20 for faster reactions
JUMP_DURATION = 0.1          # raise slightly if jumps feel incomplete
```

**Screen recorder options (free):**
- Windows: `Win + G` → Xbox Game Bar (built-in)
- OBS Studio (free, professional quality): https://obsproject.com

---

## 📹 RECORDING THE LINKEDIN VIDEO (Day 7)

**What to record:**
1. Show the terminal — AI running, printing frame predictions
2. Show Chrome window — dino jumping on its own
3. Keep recording for 60–90 seconds to show a good run
4. Optionally: show a side-by-side of terminal + game

**LinkedIn post template:**
```
I trained an AI to play Chrome Dino using only pixels from the screen.

No keyboard. No mouse. No game code access.
Just screenshots → neural network → keypresses.

Built with: Python, PyTorch, imitation learning

The AI learned by watching me play for ~30 minutes.
Then it played on its own. Here's what happened 👇

[video]

GitHub: [your repo link]

#MachineLearning #ComputerVision #Python #AI #DeepLearning
```

---

## 🐛 TROUBLESHOOTING

**"Module not found" error:**
```bash
# Make sure venv is activated
venv\Scripts\activate
pip install -r requirements.txt
```

**"Model not found" error:**
- You need to run `training/train.py` before `inference/play.py`

**AI never jumps:**
- Lower `CONFIDENCE_THRESHOLD` to 0.4 in `inference/play.py`
- Collect more jump data (let more cacti hit you while recording)
- Check class balance with `python utils/inspect_data.py`

**Screen capture is black:**
- Make sure Chrome is not minimized
- Try adjusting `CAPTURE_REGION` — run `calibrate.py` again

**pyautogui clicks in wrong place:**
- This is normal — pyautogui controls keypresses, not clicks
- Make sure Chrome window is focused/visible

**Training accuracy stuck at ~60%:**
- Collect more data (aim for 5000+ frames)
- Make sure your `CAPTURE_REGION` is actually showing the game

---

## 🔬 HOW IT WORKS (for your LinkedIn post)

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Screen  →  Capture  →  Preprocess  →  DinoNet  →  Key │
│                                                         │
│  Chrome     mss lib     80x80 gray    CNN model   space │
│  window     15 FPS      1 channel     3 outputs    /down│
│                                                         │
│              [nothing=0, jump=1, duck=2]                │
└─────────────────────────────────────────────────────────┘
```

**DinoNet architecture:**
- 4 convolutional blocks (16 → 32 → 64 → 64 filters)
- 2 fully connected layers
- ~200K parameters — tiny enough to run at 15 FPS on CPU
- Trained with cross-entropy loss + weighted sampling

---

## 🔮 NEXT STEPS (after going viral)

1. **Reinforcement Learning** — reward the AI for surviving longer, fine-tune with PPO
2. **Other games** — same pipeline works on any browser game
3. **Faster inference** — quantize the model for even lower CPU usage
4. **ONNX export** — deploy the model anywhere

---

*Built with ❤️ — inspired by PILA (PolyTrack Imitation Learning Agent)*
