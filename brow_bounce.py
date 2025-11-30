import cv2
import mediapipe as mp
import numpy as np
import random
import time

# --- GAME CONFIGURATION ---
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480
GRAVITY = 1.5
JUMP_STRENGTH = -20
GAME_SPEED = 10       # How fast obstacles move
OBSTACLE_WIDTH = 40
OBSTACLE_HEIGHT = 60
DINO_SIZE = 50
GROUND_LEVEL = 400    # Y-coordinate of the floor

# Sensitivity: Adjust this if it jumps too easily or too hard
# Higher = Harder to trigger jump
JUMP_THRESHOLD_RATIO = 1.05 
# --------------------------

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

class DinoGame:
    def __init__(self):
        self.dino_y = GROUND_LEVEL - DINO_SIZE
        self.dino_velocity = 0
        self.is_jumping = False
        self.obstacles = []
        self.score = 0
        self.game_active = False
        self.game_over = False
        
        # Calibration vars
        self.resting_brow_dist = 0
        self.calibrated = False

    def jump(self):
        if not self.is_jumping:
            self.dino_velocity = JUMP_STRENGTH
            self.is_jumping = True

    def update(self):
        if not self.game_active: return

        # Apply Gravity
        self.dino_velocity += GRAVITY
        self.dino_y += self.dino_velocity

        # Floor Collision
        if self.dino_y >= GROUND_LEVEL - DINO_SIZE:
            self.dino_y = GROUND_LEVEL - DINO_SIZE
            self.is_jumping = False
            self.dino_velocity = 0

        # Move Obstacles
        for obs in self.obstacles:
            obs[0] -= GAME_SPEED # Decrease X coordinate

        # Remove off-screen obstacles and add score
        if self.obstacles and self.obstacles[0][0] < -OBSTACLE_WIDTH:
            self.obstacles.pop(0)
            self.score += 1

        # Spawn new obstacles
        if len(self.obstacles) == 0 or self.obstacles[-1][0] < WINDOW_WIDTH - 250:
            # Random chance to spawn
            if random.randint(0, 100) < 5:
                self.obstacles.append([WINDOW_WIDTH, GROUND_LEVEL - OBSTACLE_HEIGHT])

        # Check Collisions
        dino_rect = [100, int(self.dino_y), DINO_SIZE, DINO_SIZE] # x, y, w, h
        
        for obs in self.obstacles:
            obs_rect = [obs[0], obs[1], OBSTACLE_WIDTH, OBSTACLE_HEIGHT]
            
            # AABB Collision Logic (Axis-Aligned Bounding Box)
            if (dino_rect[0] < obs_rect[0] + obs_rect[2] and
                dino_rect[0] + dino_rect[2] > obs_rect[0] and
                dino_rect[1] < obs_rect[1] + obs_rect[3] and
                dino_rect[1] + dino_rect[3] > obs_rect[1]):
                
                self.game_active = False
                self.game_over = True

    def draw(self, img):
        # Draw Floor
        cv2.line(img, (0, GROUND_LEVEL), (WINDOW_WIDTH, GROUND_LEVEL), (0, 255, 0), 5)

        # Draw Dino (Player) - Using a simple yellow square for now
        cv2.rectangle(img, 
                     (100, int(self.dino_y)), 
                     (100 + DINO_SIZE, int(self.dino_y) + DINO_SIZE), 
                     (0, 255, 255), -1) # Yellow filled

        # Draw Obstacles - Red rectangles
        for obs in self.obstacles:
            cv2.rectangle(img, 
                         (obs[0], obs[1]), 
                         (obs[0] + OBSTACLE_WIDTH, obs[1] + OBSTACLE_HEIGHT), 
                         (0, 0, 255), -1)

        # Draw Score
        cv2.putText(img, f"Score: {self.score}", (WINDOW_WIDTH - 150, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return img

# Initialize Game
game = DinoGame()
cap = cv2.VideoCapture(0)
cap.set(3, WINDOW_WIDTH)
cap.set(4, WINDOW_HEIGHT)

print("Started! Look at the camera.")
print("Press 'SPACE' to calibrate and start.")

while True:
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    # --- FACE DETECTION & JUMP LOGIC ---
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            lm = face_landmarks.landmark
            
            # Get coordinates for Right Eye and Right Eyebrow
            # ID 159: Right Eye Top
            # ID 66: Right Eyebrow Mid
            # 
            eye_top = lm[159]
            brow_mid = lm[66]
            
            # Calculate distance (only Y axis matters mostly)
            # Multiply by height to get pixels
            dist = abs(eye_top.y - brow_mid.y) * h
            
            # Draw Face Points for visual feedback
            cx_eye, cy_eye = int(eye_top.x * w), int(eye_top.y * h)
            cx_brow, cy_brow = int(brow_mid.x * w), int(brow_mid.y * h)
            cv2.circle(img, (cx_eye, cy_eye), 3, (0, 255, 0), -1)
            cv2.circle(img, (cx_brow, cy_brow), 3, (0, 255, 0), -1)
            cv2.line(img, (cx_eye, cy_eye), (cx_brow, cy_brow), (0, 255, 0), 1)

            # --- GAME TRIGGER ---
            if game.calibrated and game.game_active:
                # If current distance is significantly larger than resting distance
                if dist > game.resting_brow_dist * JUMP_THRESHOLD_RATIO:
                    game.jump()
                    cv2.putText(img, "JUMP!", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # --- CALIBRATION LOGIC ---
            if not game.calibrated:
                cv2.putText(img, "Relax face & press SPACE", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(img, f"Current Dist: {int(dist)}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                
                # Store the distance to use when space is pressed
                current_calibration_val = dist

    # --- GAME LOOP UPDATES ---
    if game.game_active:
        game.update()
        img = game.draw(img)
    elif game.game_over:
        img = game.draw(img) # Draw one last time so we see the crash
        cv2.putText(img, "GAME OVER", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.putText(img, "Press 'R' to Restart", (190, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
    
    cv2.imshow("BrowBounce", img)

    # --- CONTROLS ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' ') and not game.calibrated:
        game.resting_brow_dist = current_calibration_val
        game.calibrated = True
        game.game_active = True
        print(f"Calibrated! Resting distance: {game.resting_brow_dist}")
    elif key == ord('r') and game.game_over:
        # Reset game
        game.dino_y = GROUND_LEVEL - DINO_SIZE
        game.dino_velocity = 0
        game.obstacles = []
        game.score = 0
        game.game_active = True
        game.game_over = False

cap.release()
cv2.destroyAllWindows()