# 🤨 BrowBounce (V3)

> **A hands-free arcade platformer controlled by your face.**

**BrowBounce** turns your webcam into a game controller. Using **MediaPipe Face Mesh**, it tracks the micro-movements of your eyebrows to control a neon orb jumping over obstacles.

This project features a custom-built game engine with gravity, and collision detection.

## ✨ Features

### 🕹️ Gameplay
* **Face Control:** Jump by raising your eyebrows.
* **Dynamic Difficulty:** The game speeds up as you survive longer.
* **Persistent High Scores:** Saves your best run to disk automatically.

### 🎨 Visuals (Neon Mode)
* **Procedural Graphics:** No external assets needed—everything is drawn with code.
* **Heads-Up Display (HUD):** Features a live Speedometer, Jump Pressure Gauge, and Scoreboard.
* **Motion Trails:** The player leaves a fading cyan trail for a fluid feel.
* **Cyberpunk Aesthetic:** Darkened camera feed with Cyan/Magenta glowing elements.

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
    python brow_bounce_hud.py
    ```

## 🎮 Controls

| Key | Action |
| :--- | :--- |
| **`SPACE`** | **Calibrate & Start** (Relax face first!) |
| **`Eyebrows`**| **JUMP** (Raise them high!) |
| **`W`** | **Decrease Sensitivity** (Make it harder to jump) |
| **`S`** | **Increase Sensitivity** (Make it easier to jump) |
| **`R`** | **Restart** (After Game Over) |
| **`Q`** | **Quit** |

## ⚙️ How it Works

1.  **Face Tracking:** Calculates the pixel distance between your **Nose Tip** (Landmark 1) and **Mid-Eyebrow** (Landmark 66).
2.  **Calibration:** When you press SPACE, it records your "Resting Distance."
3.  **Trigger Logic:** If `Current Distance > Resting Distance * Sensitivity`, a jump occurs.
4.  **Rendering:** Uses `cv2.addWeighted` to create transparent "glass" overlays for the UI panels.

## 📂 Files
* `brow_bounce_hud.py`: The main game script.
* `highscore.txt`: Automatically created to save your best score.

## 🤝 Contributing
Pull requests are welcome! Ideas for future updates:
* Add power-ups (slow motion, shields).
* Add sound effects (using `playsound` library).

## 📜 License
MIT License.