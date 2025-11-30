import cv2
import mediapipe as mp
import numpy as np
import random
import os

# --- CONFIGURATION ---
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480
GROUND_LEVEL = 400
GRAVITY = 1.2
JUMP_POWER = -18
START_SPEED = 8
MAX_SPEED = 20

# Initial Sensitivity (1.05 = 5% brow lift triggers jump)
DEFAULT_SENSITIVITY = 1.05 

# Visuals
BALL_RADIUS = 15
BALL_COLOR = (0, 255, 255) # Yellow

# Save File
HIGHSCORE_FILE = "highscore.txt"
# ---------------------

def overlay_image(background, overlay, x, y, size_w, size_h):
    if overlay is None: return background
    overlay_resized = cv2.resize(overlay, (size_w, size_h))
    bg_h, bg_w, _ = background.shape
    ol_h, ol_w, _ = overlay_resized.shape
    if x >= bg_w or y >= bg_h: return background
    if x + ol_w > bg_w: ol_w = bg_w - x
    if y + ol_h > bg_h: ol_h = bg_h - y
    if x < 0: ol_w += x; x = 0
    if y < 0: ol_h += y; y = 0
    if ol_w <= 0 or ol_h <= 0: return background
    roi = background[y:y+ol_h, x:x+ol_w]
    overlay_crop = overlay_resized[:ol_h, :ol_w]
    if overlay_crop.shape[2] == 4:
        alpha = overlay_crop[:, :, 3] / 255.0
        alpha = np.dstack([alpha]*3)
        fg = overlay_crop[:, :, :3] * alpha
        bg = roi * (1.0 - alpha)
        combined = cv2.add(fg, bg)
        background[y:y+ol_h, x:x+ol_w] = combined
    else:
        background[y:y+ol_h, x:x+ol_w] = overlay_crop
    return background

class ArcadeGame:
    def __init__(self):
        self.dino_y = GROUND_LEVEL - (BALL_RADIUS * 2)
        self.velocity = 0
        self.is_jumping = False
        self.obstacles = []
        self.score = 0
        self.speed = START_SPEED
        self.active = False
        self.game_over = False
        self.calibrated = False
        self.resting_dist = 0
        self.sensitivity = DEFAULT_SENSITIVITY
        
        # Load High Score from file
        self.high_score = self.load_high_score()
        
        self.img_cactus = cv2.imread("assets/cactus.png", cv2.IMREAD_UNCHANGED)

    def load_high_score(self):
        if not os.path.exists(HIGHSCORE_FILE):
            return 0
        try:
            with open(HIGHSCORE_FILE, "r") as f:
                return int(f.read())
        except:
            return 0

    def save_high_score(self):
        with open(HIGHSCORE_FILE, "w") as f:
            f.write(str(self.high_score))

    def reset(self):
        self.dino_y = GROUND_LEVEL - (BALL_RADIUS * 2)
        self.velocity = 0
        self.obstacles = []
        self.score = 0
        self.speed = START_SPEED
        self.active = True
        self.game_over = False

    def jump(self):
        if not self.is_jumping:
            self.velocity = JUMP_POWER
            self.is_jumping = True

    def update(self):
        if not self.active: return
        self.velocity += GRAVITY
        self.dino_y += self.velocity
        
        # Floor collision
        if self.dino_y >= GROUND_LEVEL - (BALL_RADIUS * 2):
            self.dino_y = GROUND_LEVEL - (BALL_RADIUS * 2)
            self.is_jumping = False
            self.velocity = 0
            
        # Move Obstacles
        for obs in self.obstacles:
            obs[0] -= self.speed
            
        # Clean Obstacles & Score
        if self.obstacles and self.obstacles[0][0] < -50:
            self.obstacles.pop(0)
            self.score += 1
            if self.score % 5 == 0 and self.speed < MAX_SPEED:
                self.speed += 1
                
        # Spawn logic
        if len(self.obstacles) == 0 or self.obstacles[-1][0] < WINDOW_WIDTH - random.randint(300, 500):
            self.obstacles.append([WINDOW_WIDTH, GROUND_LEVEL - 60])
            
        # Collision Logic
        ball_center_x = 80 + BALL_RADIUS
        ball_center_y = int(self.dino_y) + BALL_RADIUS
        
        for obs in self.obstacles:
            ox, oy, ow, oh = int(obs[0]), int(obs[1]), 40, 60
            
            if (ball_center_x + BALL_RADIUS > ox and 
                ball_center_x - BALL_RADIUS < ox + ow and 
                ball_center_y + BALL_RADIUS > oy and 
                ball_center_y - BALL_RADIUS < oy + oh):
                
                self.active = False
                self.game_over = True
                
                # Check and Save High Score
                if self.score > self.high_score:
                    self.high_score = self.score
                    self.save_high_score()
                    print(f"New High Score Saved: {self.high_score}")

    def draw(self, img):
        cv2.line(img, (0, GROUND_LEVEL), (WINDOW_WIDTH, GROUND_LEVEL), (200, 200, 200), 2)

        # Draw Ball
        center_x = 80 + BALL_RADIUS
        center_y = int(self.dino_y) + BALL_RADIUS
        cv2.circle(img, (center_x, center_y), BALL_RADIUS, BALL_COLOR, -1)
        cv2.circle(img, (center_x, center_y), BALL_RADIUS, (0,0,0), 2)

        # Draw Obstacles
        for obs in self.obstacles:
            if self.img_cactus is not None:
                img = overlay_image(img, self.img_cactus, int(obs[0]), int(obs[1]), 40, 60)
            else:
                cv2.rectangle(img, (int(obs[0]), int(obs[1])), (int(obs[0])+40, int(obs[1])+60), (0, 0, 255), -1)

        # Draw Scores
        # Current Score (Big, Top Right)
        cv2.putText(img, f"{self.score}", (WINDOW_WIDTH-80, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        # High Score (Small, below current)
        cv2.putText(img, f"HI: {self.high_score}", (WINDOW_WIDTH-130, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Draw Sensitivity Debugger (Bottom Right)
        sens_text = f"SENS: {self.sensitivity:.2f}"
        cv2.putText(img, sens_text, (WINDOW_WIDTH-140, WINDOW_HEIGHT-20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
        cv2.putText(img, "[W / S] to adjust", (WINDOW_WIDTH-180, WINDOW_HEIGHT-40), cv2.FONT_HERSHEY_PLAIN, 0.8, (200, 200, 200), 1)
        
        return img

# Initialize
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True)
game = ArcadeGame()

cap = cv2.VideoCapture(0)
cap.set(3, WINDOW_WIDTH)
cap.set(4, WINDOW_HEIGHT)

print("BrowBounce Final Version Loaded.")
print("Use 'W' (Harder) and 'S' (Easier) to tune sensitivity.")

while True:
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    
    current_dist = 0

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            pt_brow = landmarks.landmark[66]
            pt_nose = landmarks.landmark[1]
            
            cx_brow, cy_brow = int(pt_brow.x * w), int(pt_brow.y * h)
            cx_nose, cy_nose = int(pt_nose.x * w), int(pt_nose.y * h)
            
            current_dist = abs(cy_nose - cy_brow)

    # --- LOGIC ---
    if game.calibrated:
        threshold = game.resting_dist * game.sensitivity
        
        # Jump Bar
        bar_height = int(current_dist * 2)
        thresh_height = int(threshold * 2)
        
        cv2.line(img, (10, 400 - thresh_height), (30, 400 - thresh_height), (0, 0, 255), 2)
        color = (255, 255, 255)
        if current_dist > threshold:
            color = (0, 255, 0)
        cv2.rectangle(img, (15, 400 - bar_height), (25, 400), color, -1)

        if current_dist > threshold:
             game.jump()

        if game.active:
            game.update()
        elif game.game_over:
            overlay = img.copy()
            cv2.rectangle(overlay, (150, 150), (490, 330), (0, 0, 0), -1)
            img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
            cv2.putText(img, "GAME OVER", (190, 210), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 255), 2)
            cv2.putText(img, f"Score: {game.score}", (260, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
            cv2.putText(img, "Press [R] to Retry", (210, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    else:
        cv2.putText(img, "RELAX FACE", (220, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, "PRESS SPACE", (210, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, f"Raw Dist: {int(current_dist)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 1)

    img = game.draw(img)
    cv2.imshow("BrowBounce Final", img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        if not game.calibrated:
            game.resting_dist = current_dist
            game.calibrated = True
            game.reset()
    elif key == ord('r') and game.game_over:
        game.reset()
    
    # Tuning Controls (W/S are safer than arrows)
    elif key == ord('w'): # Harder
        game.sensitivity += 0.01
    elif key == ord('s'): # Easier
        game.sensitivity -= 0.01
        if game.sensitivity < 1.01: game.sensitivity = 1.01

cap.release()
cv2.destroyAllWindows()