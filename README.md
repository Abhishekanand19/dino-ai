# 🦕 Chrome Dino AI — Vision-Based Imitation Learning

An AI agent that learns to play Chrome Dino directly from screen pixels using Computer Vision and Imitation Learning.

The agent observes gameplay frames, learns from human demonstrations, and autonomously controls the Dino without accessing game internals.

---

## Project Overview

This project implements a complete machine learning pipeline:

1. Screen capture and gameplay recording
2. Dataset creation from human demonstrations
3. CNN-based behavioral cloning
4. Real-time inference and autonomous gameplay

The AI learns the relationship between visual game states and player actions (jump or do nothing) and then attempts to replicate the learned behavior.

---

## Results

### Dataset

* Total Frames: **7,576**
* Gameplay Sessions: **8**
* Jump Examples: **1,106**
* No Missing Frames

### Model Performance

* Validation Accuracy: **90.9%**
* Framework: **PyTorch**
* Approach: **Behavioral Cloning / Imitation Learning**

### Autonomous Gameplay

* Average Score Across 5 Runs: **145**
* Stable autonomous obstacle avoidance
* Real-time CPU inference

---

## System Architecture

```text
Human Gameplay
       ↓
Screen Capture
       ↓
Dataset Creation
       ↓
CNN Training
       ↓
Model Inference
       ↓
Autonomous Dino Control
```

---

## Project Structure

```text
dino-ai/
│
├── data_collection/
│   ├── calibrate.py
│   └── collect.py
│
├── training/
│   └── train.py
│
├── inference/
│   └── play.py
│
├── utils/
│   ├── model.py
│   └── inspect_data.py
│
├── models/
│   └── dino_model.pth
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Technologies Used

* Python
* PyTorch
* OpenCV
* NumPy
* MSS
* PyAutoGUI
* Computer Vision
* Imitation Learning

---

## How It Works

### Data Collection

The system records gameplay frames while a human plays Chrome Dino.

Each frame is paired with the corresponding player action:

* Nothing
* Jump

This creates a supervised learning dataset.

### Model Training

A lightweight Convolutional Neural Network (CNN) is trained to map:

```text
Game Frame → Player Action
```

The model learns visual patterns associated with obstacles and jump timing.

### Autonomous Play

During inference:

1. Capture screen
2. Preprocess frame
3. Predict action
4. Execute key press
5. Repeat in real time

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/chrome-dino-ai.git
cd chrome-dino-ai
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

Collect gameplay data:

```bash
python data_collection/collect.py
```

Inspect dataset:

```bash
python utils/inspect_data.py
```

Train model:

```bash
python training/train.py
```

---

## Run the AI

```bash
python inference/play.py
```

Open Chrome Dino, keep the game visible, and allow the AI to take control.

---

## Key Learning Outcomes

This project demonstrates:

* Computer Vision
* Data Collection Pipelines
* Deep Learning with CNNs
* Behavioral Cloning
* Real-Time Inference
* Human Demonstration Learning
* End-to-End Machine Learning Deployment

---

## Future Improvements

* Reinforcement Learning (PPO)
* Duck Detection and Control
* ONNX Model Export
* Live Prediction Overlay
* Model Quantization
* Higher FPS Inference

---

## Author

Abhishek Anand
If you found this project interesting, feel free to star the repository.
