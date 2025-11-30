import cv2
import mediapipe as mp
import numpy as np
import random
import os
from collections import deque

# --- CONFIGURATION ---
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480
GROUND_LEVEL = 400
GRAVITY = 1.2
JUMP_POWER = -18
START_SPEED = 8
MAX_SPEED = 20
DEFAULT_SENSITIVITY = 1.05 
BALL_RADIUS = 15

# --- COLORS (B, G, R) ---
C_BG_DIM = 0.5         # 50% darkness for camera
C_CYAN = (255, 255, 0) # Player / Borders
C_MAGENTA = (203, 192, 255) # Enemy
C_GREEN = (0, 255, 0)  # Ground
C_WHITE = (255, 255, 255)
C_DARK = (30, 30, 30)  # UI Backgrounds
# ---------------------

class ArcadeGame:
    def __init__(self):
        self.dino_y = GROUND_LEVEL - (BALL_RADIUS * 2)
        self.velocity = 0
        self.is_jumping = False
        self.obstacles = []
        self.score = 0
        self.high_score = self.load_high_score()
        self.speed = START_SPEED
        self.active = False
        self.game_over = False
        self.calibrated = False
        self.resting_dist = 0
        self.sensitivity = DEFAULT_SENSITIVITY
        self.trail = deque(maxlen=8)

    def load_high_score(self):
        if not os.path.exists("highscore.txt"): return 0
        try:
            with open("highscore.txt", "r") as f: return int(f.read())
        except: return 0

    def save_high_score(self):
        with open("highscore.txt", "w") as f: f.write(str(self.high_score))

    def reset(self):
        self.dino_y = GROUND_LEVEL - (BALL_RADIUS * 2)
        self.velocity = 0
        self.obstacles = []
        self.score = 0
        self.speed = START_SPEED
        self.active = True
        self.game_over = False
        self.trail.clear()

    def jump(self):
        if not self.is_jumping:
            self.velocity = JUMP_POWER
            self.is_jumping = True

    def update(self):
        if not self.active: return
        self.velocity += GRAVITY
        self.dino_y += self.velocity
        if self.dino_y >= GROUND_LEVEL - (BALL_RADIUS * 2):
            self.dino_y = GROUND_LEVEL - (BALL_RADIUS * 2)
            self.is_jumping = False
            self.velocity = 0
        
        # Trail
        center_x = 100
        center_y = int(self.dino_y) + BALL_RADIUS
        self.trail.appendleft((center_x, center_y))

        # Obstacles
        for obs in self.obstacles:
            obs[0] -= self.speed
        if self.obstacles and self.obstacles[0][0] < -50:
            self.obstacles.pop(0)
            self.score += 1
            # Speed up
            if self.score % 5 == 0 and self.speed < MAX_SPEED:
                self.speed += 1
                
        if len(self.obstacles) == 0 or self.obstacles[-1][0] < WINDOW_WIDTH - random.randint(300, 500):
            self.obstacles.append([WINDOW_WIDTH, GROUND_LEVEL])
            
        # Collision
        for obs in self.obstacles:
            ox, oy = int(obs[0]), int(obs[1])
            dist = np.hypot(center_x - (ox + 20), center_y - (oy - 30))
            if dist < BALL_RADIUS + 25:
                self.active = False
                self.game_over = True
                if self.score > self.high_score:
                    self.save_high_score()
                    self.high_score = self.score

    def draw_hud_panel(self, img, x, y, w, h, title, value, color_border):
        """Helper to draw a glass-like panel"""
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x+w, y+h), C_DARK, -1)
        # Apply transparency (Glass effect)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        # Border
        cv2.rectangle(img, (x, y), (x+w, y+h), color_border, 2)
        # Text
        cv2.putText(img, title, (x+10, y+20), cv2.FONT_HERSHEY_PLAIN, 1, C_WHITE, 1)
        cv2.putText(img, value, (x+10, y+55), cv2.FONT_HERSHEY_TRIPLEX, 1.2, C_WHITE, 2)

    def draw(self, img):
        # 1. Darken Background
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (WINDOW_WIDTH, WINDOW_HEIGHT), (20, 20, 40), -1)
        img = cv2.addWeighted(overlay, 1 - C_BG_DIM, img, C_BG_DIM, 0)

        # 2. Draw Ground
        cv2.line(img, (0, GROUND_LEVEL), (WINDOW_WIDTH, GROUND_LEVEL), C_GREEN, 2)
        cv2.line(img, (0, GROUND_LEVEL+4), (WINDOW_WIDTH, GROUND_LEVEL+4), (0, 100, 0), 4)

        # 3. Draw Player & Trail
        for i, pos in enumerate(self.trail):
            radius = int(BALL_RADIUS - (i * 1.5))
            if radius > 0: cv2.circle(img, pos, radius, (200, 200, 0), -1)

        center_x = 100
        center_y = int(self.dino_y) + BALL_RADIUS
        cv2.circle(img, (center_x, center_y), BALL_RADIUS + 4, (255, 255, 100), -1) # Glow
        cv2.circle(img, (center_x, center_y), BALL_RADIUS, C_CYAN, -1)

        # 4. Draw Obstacles (Triangles)
        for obs in self.obstacles:
            x, y = int(obs[0]), int(obs[1])
            pt1 = (x + 20, y - 60)
            pt2 = (x, y); pt3 = (x + 40, y)
            cnt = np.array([pt1, pt2, pt3])
            cv2.drawContours(img, [cnt], 0, C_MAGENTA, -1)
            cv2.drawContours(img, [cnt], 0, C_WHITE, 2)

        # --- 5. THE NEW UI DASHBOARD ---
        
        # A. Top Glass Bar
        bar_overlay = img.copy()
        cv2.rectangle(bar_overlay, (0, 0), (WINDOW_WIDTH, 80), C_DARK, -1)
        cv2.addWeighted(bar_overlay, 0.7, img, 0.3, 0, img)
        cv2.line(img, (0, 80), (WINDOW_WIDTH, 80), C_CYAN, 2)

        # B. Score Panels (Left and Right)
        # Current Score
        cv2.putText(img, "SCORE", (30, 30), cv2.FONT_HERSHEY_PLAIN, 1, C_CYAN, 1)
        cv2.putText(img, f"{self.score:03d}", (30, 70), cv2.FONT_HERSHEY_TRIPLEX, 1.5, C_WHITE, 2)
        
        # High Score
        cv2.putText(img, "BEST", (200, 30), cv2.FONT_HERSHEY_PLAIN, 1, (200, 200, 200), 1)
        cv2.putText(img, f"{self.high_score:03d}", (200, 70), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (200, 200, 200), 2)

        # C. Speedometer (Right Side)
        cv2.putText(img, "SPEED", (WINDOW_WIDTH-150, 30), cv2.FONT_HERSHEY_PLAIN, 1, C_MAGENTA, 1)
        # Background bar
        cv2.rectangle(img, (WINDOW_WIDTH-150, 45), (WINDOW_WIDTH-30, 65), (50, 50, 50), -1)
        # Fill bar based on speed (Map speed 8-20 to width)
        fill_percent = (self.speed - START_SPEED) / (MAX_SPEED - START_SPEED)
        fill_width = int(120 * fill_percent)
        cv2.rectangle(img, (WINDOW_WIDTH-150, 45), (WINDOW_WIDTH-150+fill_width, 65), C_MAGENTA, -1)
        cv2.rectangle(img, (WINDOW_WIDTH-150, 45), (WINDOW_WIDTH-30, 65), C_WHITE, 1) # Border

        return img

    def draw_jump_gauge(self, img, current_dist, threshold):
        """Draws a vertical pressure gauge on the left"""
        x, y, w, h = 20, 150, 30, 200
        
        # Background
        cv2.rectangle(img, (x, y), (x+w, y+h), (20, 20, 20), -1)
        cv2.rectangle(img, (x, y), (x+w, y+h), C_CYAN, 1)
        
        # Fill Level
        # Max reasonable distance is roughly 2x threshold for visualization
        max_val = threshold * 2 if threshold > 0 else 100
        fill_h = int((current_dist / max_val) * h)
        if fill_h > h: fill_h = h
        
        # Color changes if jump triggered
        c_fill = C_WHITE
        if current_dist > threshold: c_fill = C_CYAN
            
        # Draw Fill (Bottom up)
        cv2.rectangle(img, (x+2, y+h-fill_h), (x+w-2, y+h), c_fill, -1)
        
        # Draw Threshold Line (The "Target")
        thresh_y = y + h - int((threshold / max_val) * h)
        cv2.line(img, (x-5, thresh_y), (x+w+5, thresh_y), C_MAGENTA, 2)
        cv2.putText(img, "JUMP", (x+35, thresh_y+5), cv2.FONT_HERSHEY_PLAIN, 0.8, C_MAGENTA, 1)

# --- SETUP ---
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True)
game = ArcadeGame()

cap = cv2.VideoCapture(0)
cap.set(3, WINDOW_WIDTH)
cap.set(4, WINDOW_HEIGHT)

print("BrowBounce HUD Loaded.")

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
            cx_nose, cy_nose = int(pt_nose.x * w), int(pt_nose.y * h)
            current_dist = abs(cy_nose - int(pt_brow.y * h))
            
            if not game.calibrated:
                cv2.circle(img, (cx_nose, cy_nose), 5, C_CYAN, -1)

    # --- GAME LOOP ---
    if game.calibrated:
        threshold = game.resting_dist * game.sensitivity
        
        # LOGIC
        if current_dist > threshold: game.jump()
        if game.active: game.update()

        # DRAW SCENE
        img = game.draw(img)
        
        # DRAW GAUGE (Function separated for cleaner code)
        game.draw_jump_gauge(img, current_dist, threshold)
        
        # GAME OVER OVERLAY
        if game.game_over:
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 150), (WINDOW_WIDTH, 330), (0, 0, 0), -1)
            img = cv2.addWeighted(overlay, 0.85, img, 0.15, 0)
            cv2.rectangle(img, (0, 150), (WINDOW_WIDTH, 330), C_MAGENTA, 3) # Border
            
            cv2.putText(img, "SYSTEM FAILURE", (140, 220), cv2.FONT_HERSHEY_TRIPLEX, 1.5, C_MAGENTA, 2)
            cv2.putText(img, f"FINAL SCORE: {game.score}", (220, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_WHITE, 1)
            cv2.putText(img, "PRESS [R] TO REBOOT", (210, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_CYAN, 1)
            
    else:
        # START SCREEN
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (WINDOW_WIDTH, WINDOW_HEIGHT), (10, 10, 20), -1)
        img = cv2.addWeighted(overlay, 0.8, img, 0.2, 0)
        
        cv2.putText(img, "NEON JUMP", (140, 200), cv2.FONT_HERSHEY_TRIPLEX, 2.5, C_CYAN, 3)
        cv2.putText(img, "INITIALIZE: PRESS SPACE", (160, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_WHITE, 1)

    cv2.imshow("BrowBounce HUD", img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord(' '):
        if not game.calibrated:
            game.resting_dist = current_dist
            game.calibrated = True
            game.reset()
    elif key == ord('r') and game.game_over:
        game.reset()
    elif key == ord('w'): game.sensitivity += 0.01
    elif key == ord('s'): 
        game.sensitivity -= 0.01
        if game.sensitivity < 1.01: game.sensitivity = 1.01

cap.release()
cv2.destroyAllWindows()