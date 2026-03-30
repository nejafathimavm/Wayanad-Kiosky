# cv2 - Used for camera and image processing 
import cv2
# mediapipe - Used for hand tracking
import mediapipe as mp
# mp_python - MediaPipe task API
from mediapipe.tasks import python as mp_python
# vision - Used for vision models like hand detection
from mediapipe.tasks.python import vision
# time - Used for timestamps
import time
# os - Checks if files exist
import os
# urllib.request - Downloads files from internet
import urllib.request

# ── Download the hand detection model ──────────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task" # Stores the model file name
if not os.path.exists(MODEL_PATH): # Checks if the model file is already in the folder
    print("Downloading hand landmarker model (~25 MB)...")
    url = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    ) # URL of the hand tracking AI model
    urllib.request.urlretrieve(url, MODEL_PATH) # Downloads the model
    print("Download complete.")

# ── Initialize handLandmarker (Tasks API) ────────────────────────────────────────────────
base_options  = mp_python.BaseOptions(model_asset_path=MODEL_PATH) # Loads the downloaded model
options       = vision.HandLandmarkerOptions(  # Creates configuration settings
    base_options = base_options,
    num_hands    = 1, # Detects only one hand
    running_mode = vision.RunningMode.VIDEO,   # Processes frames one by one from video
)

# ── Application variables ─────────────────────────────────────────────────────────────────
canvas       = None # Canvas stores the drawing layer 
active_color = (255, 0, 0)   # default: blue (BGR)

# Create color palette boxes: (x1, y1, x2, y2, BGR-color)
# Creates 4 color selection boxes
# Coordinates of rectangles
palette = [
    (20,  20, 70,  70, (0,   0,   255)),  # Red
    (80,  20, 130, 70, (0,   255, 0  )),  # Green
    (140, 20, 190, 70, (255, 0,   0  )),  # Blue
    (200, 20, 250, 70, (0,   255, 255)),  # Yellow
]

# ── Function to draw palette ───────────────────────────────────────────────────────────────────
def draw_palette(frame, active_col): # Creates a function to draw color boxes
    """Draw color swatches; highlight the active one."""
    for x1, y1, x2, y2, color in palette:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1) # Draws filled rectangles
        if color == active_col: # Checks the selected color
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 3) # Draws white border around active color 

# Function to draw finger landmarks
def draw_landmarks(frame, lmList): # Function to show fingertip points
    """Draw circles on key fingertip landmarks."""
    for tip_idx in [4, 8, 12, 16, 20]: # These numbers represent fingertip landmarks
        cv2.circle(frame, lmList[tip_idx], 6, (255, 255, 255), -1) # Draw circles on fingertips
    cv2.circle(frame, lmList[8], 8, (0, 200, 255), -1) # Highlights index finger

# ── Main loop ─────────────────────────────────────────────────────────────────
cap        = cv2.VideoCapture(0) # Opens webcam
start_time = time.time() # Stores starting time

with vision.HandLandmarker.create_from_options(options) as landmarker: # Loads the hand detection model
    while True: # Runs program continuously
        ret, frame = cap.read() # Reads image from camera
        if not ret:
            break

        frame = cv2.flip(frame, 1) # Flips frame horizontally like a mirror

        if canvas is None:              # Creates empty drawing board
            canvas = frame.copy() * 0   # blank canvas same size as frame

        # Convert BGR frame → MediaPipe Image
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # MediaPipe requires RGB format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb) # Converts frame in to MediaPipe image

        # Detect hand (VIDEO mode requires monotonically increasing timestamps)
        timestamp_ms = int((time.time() - start_time) * 1000) # Create timestamp
        results      = landmarker.detect_for_video(mp_image, timestamp_ms) # Detects hand landmarks

        # Draw UI elements
        draw_palette(frame, active_color) # Shows color palette
        cv2.putText(frame, "C = clear | ESC = quit", # Displays instructions
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if results.hand_landmarks: # Check if hand detected
            h, w, _ = frame.shape # Gets frame dimensions
            lm      = results.hand_landmarks[0] 
            lmList  = [(int(l.x * w), int(l.y * h)) for l in lm] # Converts normalized coordinates to pixel coordinates

            draw_landmarks(frame, lmList)

            # Detect finger-up states (tip y < pip y  →  finger is extended)
            thumb_up  = lmList[4][0]  > lmList[3][0] # Checks thumb position
            index_up  = lmList[8][1]  < lmList[6][1]
            middle_up = lmList[12][1] < lmList[10][1]
            ring_up   = lmList[16][1] < lmList[14][1]
            pinky_up  = lmList[20][1] < lmList[18][1] # If tip is above joint - finger is up

            # -- Clear All Mode(5 fingers) -----------------------------
            if thumb_up and index_up and middle_up and ring_up and pinky_up: # If all fingers are raised
                canvas = frame.copy()*0 # Clears drawing
                cv2.putText(frame,"Clear All",(50,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            
            # ── Eraser: index + middle + ring up, pinky down ──────────────
            elif index_up and middle_up and ring_up and not pinky_up:
                cv2.circle(canvas, lmList[8], 30, (0, 0, 0), -1) # Draw black circle to erase
                cv2.putText(frame, "Eraser Mode", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)

            # ── Selection: index + middle up only ────────────────────────
            elif index_up and middle_up and not (ring_up or pinky_up):
                x, y = lmList[8]
                for x1, y1, x2, y2, color in palette:
                    if x1 < x < x2 and y1 < y < y2:
                        active_color = color  # checks if finger touches palette box and changes color
                cv2.putText(frame, "Selection Mode", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # ── Drawing: index only ───────────────────────────────────────
            elif index_up and not (middle_up or ring_up or pinky_up):
                cv2.circle(canvas, lmList[8], 8, active_color, -1) # Draws colored circle
                cv2.putText(frame, "Drawing Mode", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, active_color, 2)

        # Merge drawing with camera
        output = cv2.addWeighted(frame, 1, canvas, 0.5, 0) # Combines canvas and video frame
        cv2.imshow("Air Drawing  |  Tasks API", output) # Displays output on the window

        key = cv2.waitKey(1) & 0xFF # Detects keyboard input
        if key == 27:          # ESC key → exit
            break
        elif key == ord('c'):  # Press C   → clear canvas
            canvas = frame.copy() * 0

cap.release() # Stops webcam
cv2.destroyAllWindows() # Close all windows