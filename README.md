# 🤨 BrowBounce (V1)

> **The game where you jump by raising your eyebrows.**

**BrowBounce** is a hands-free arcade game powered by Computer Vision. It uses **MediaPipe Face Mesh** to track the distance between your eye and eyebrow in real-time. When you look surprised, the character jumps!

## ✨ Features
* **Face Control:** No keyboard needed—just your facial expressions.
* **Auto-Calibration:** Adjusts to your specific face shape.
* **Real-time Physics:** Gravity and collision detection built from scratch.

## 🛠️ Installation

1.  **Clone the repo**
    ```bash
    git clone https://github.com/thanyow/BrowBounce.git
    cd BrowBounce
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the game**
    ```bash
    python brow_bounce.py
    ```

## 🎮 How to Play

1.  **Launch the game.** You will see your camera feed with green dots on your face.
2.  **CALIBRATE:** Relax your face (don't smile or frown) and press **SPACE**.
3.  **PLAY:** The yellow box will start running.
4.  **JUMP:** Raise your eyebrows high (look surprised!) to jump over the red obstacles.
5.  **RESTART:** If you crash, press **'R'** to try again.

## ⚙️ How it Works
The script calculates the vertical distance between **Face Landmark 159** (Top of Eye) and **Landmark 66** (Eyebrow). If this distance exceeds your calibrated "resting" distance by a specific threshold (1.05x), a jump is triggered.

## 📜 License
MIT License.