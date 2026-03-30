# import libraries
import cv2 # OpenCV library for webcam capture and drawing UI
import mediapipe as mp # MediaPipe library for AI hand detection 
import time # Time module for timers and delays
import numpy as np # Numpy for image arrays and numerical processing
import pyttsx3 # Text to speech engine for voice output
import os # Used for file and folder operations.
import random # Random module for generating dummy weather
import qrcode # Library to generate QR codes
import urllib.request # Used to download files from internet
from collections import deque # Efficient list for storing cursor trail

# ── Download the hand detection model ──────────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task" # Stores the model file name
if not os.path.exists(MODEL_PATH): # Checks if the model file is already in the folder
    print("Downloading hand landmarker model (~25 MB)...")
    url = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    ) # URL of the hand tracking AI model
    urllib.request.urlretrieve(url, MODEL_PATH) # Downloads the model
    print("Download complete.")


# -------- INITIAL SETUP --------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # Get the directory path of the current Python script
engine = pyttsx3.init() # Initialise the text-to-speech engine


# SET TO FEMALE VOICE
voices = engine.getProperty('voices') # Get list of voices installed in the system
for voice in voices: # Loop through voices to find a female voice
    if "female" in voice.name.lower() or "zira" in voice.name.lower(): # Zira is common on Windows
        engine.setProperty('voice', voice.id)
        break

engine.setProperty('rate', 170) # Set speaking speed

# Utility to make the AI speak
def speak(text):
    engine.say(text) # Convert text to speech
    engine.runAndWait() # Wait until speaking is finished

# MediaPipe AI Setup 
BaseOptions = mp.tasks.BaseOptions # Base configuration class
HandLandmarker = mp.tasks.vision.HandLandmarker # Main class used for detecting hands
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = os.path.join(SCRIPT_DIR, 'hand_landmarker.task')
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'), # Load the downloaded model file
    running_mode=VisionRunningMode.VIDEO, num_hands=1 # Set detection mode to video
)
landmarker = HandLandmarker.create_from_options(options) # Create the hand detection object

WIDTH, HEIGHT = 1280, 720 # 1280 pixels width # 720 pixels height

# -------- WAYANAD DATA HUB --------
# Tourist attractions data
ATTRACTIONS = {
    "Edakkal Caves": {
        "desc": "Ancient rock engravings from the Neolithic age.",
        "act": "Trekking, History Walk", "time": "9:30 AM-4 PM", "fee": "Rs 50",
        "map": "https://maps.app.goo.gl/Edakkal", "folder": "edakkal"
    },
    "Banasura Dam": {
        "desc": "India's largest earth dam.",
        "act": "Speed Boating, Trekking", "time": "9 AM-5 PM", "fee": "Rs 40",
        "map": "https://maps.app.goo.gl/Banasura", "folder": "banasura"
    },
    "Pookode Lake": {
        "desc": "Freshwater lake in evergreen forests.",
        "act": "Boating, Spices Shopping", "time": "9 AM-5 PM", "fee": "Rs 20",
        "map": "https://maps.app.goo.gl/Pookode", "folder": "pookode"
    },
    "Soochipara": {
        "desc": "Spectacular three-tiered waterfall.",
        "act": "Nature Walk, Photography", "time": "9 AM-4:30 PM", "fee": "Rs 80",
        "map": "https://maps.app.goo.gl/Soochipara", "folder": "soochipara"
    },
    "Chembra Peak": {
        "desc": "Highest peak in Wayanad with a heart-shaped lake.",
        "act": "Trekking, Peak Views", "time": "7 AM-2 PM", "fee": "Rs 20",
        "map": "https://maps.app.goo.gl/Chembra", "folder": "chembra"
    },
    "Kuruva Island": {
        "desc": "River delta with dense flora and fauna.",
        "act": "Bamboo Rafting, Nature Trail", "time": "9 AM-3:30 PM", "fee": "Rs 110",
        "map": "https://maps.app.goo.gl/Kuruva", "folder": "kuruva"
    }
}

# Dummy Weather Function
def get_weather(location):# In a real app, use requests.get(f"http://api.openweathermap.org/data/2.5/weather?q={location}...")
    temp = random.randint(22, 28)
    condition = random.choice(["Sunny", "Cloudy", "Mist"])
    return f"{temp}C | {condition}"

# -------- LOAD IMAGES AND GENERATE QR --------
IMAGES = {}
QR_CODES = {}

for name, info in ATTRACTIONS.items():
    IMAGES[name] = []
    path = os.path.join(SCRIPT_DIR, "images", info["folder"]) # Create path to image folder
    if os.path.exists(path):
        for f in os.listdir(path):
            if f.lower().endswith((".jpg", ".png")):
                img = cv2.imread(os.path.join(path, f))
                if img is not None:
                    IMAGES[name].append(cv2.resize(img, (400, 250))) # Resize images for slideshow
    if not IMAGES[name]:
        IMAGES[name].append(np.zeros((250, 400, 3), np.uint8)) # If no image found, create blank placeholder

    qr = qrcode.QRCode(box_size=5) # Create QR code
    qr.add_data(info["map"]) # Add google map link
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white").convert('RGB')
    QR_CODES[name] = cv2.resize(np.array(qr_img), (150, 150))

bg = cv2.resize(cv2.imread(os.path.join(SCRIPT_DIR, "images/wayanad_bg.jpg")), (WIDTH, HEIGHT)) # load background image and resize to screen

# -------- MAIN KIOSK CLASS --------
class WayanadUltimateKiosk:
    def __init__(self):
        self.state = "IDLE" # Current screen state
        self.selected = "" # Selected tourist place
        self.last_active = time.time() # Time of last user interaction
        self.img_index = 0 # Slideshow index
        self.last_img_change = time.time() 
        self.trail_points = deque(maxlen=15) # Cursor trail points
        self.intro_played = False

    def run(self):
        cap = cv2.VideoCapture(0) # Start webcam
        
        # --- STARTUP VOICE ---
        print("Kiosk Initialized. Ready for interaction.")

        while cap.isOpened():
            success, raw_frame = cap.read()
            if not success: break
            
            frame = bg.copy() # Copy background image
            raw_frame = cv2.flip(raw_frame, 1) # mirror camera image
            raw_frame = cv2.resize(raw_frame, (WIDTH, HEIGHT))
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=raw_frame), int(time.time()*1000)) # Detect hand landmarks


            # IDLE SCREEN
            if self.state == "IDLE":
                text = "WELCOME TO WAYANAD - RAISE HAND TO START"
                font = cv2.FONT_HERSHEY_DUPLEX
                scale = 1.2
                thickness = 2
                (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
                text_x = (WIDTH - text_w) // 2
                text_y = (HEIGHT + text_h) // 2
                cv2.putText(frame, text, (text_x + 2, text_y + 2), font, scale, (0, 0, 0), thickness + 2)
                cv2.putText(frame, text, (text_x, text_y), font, scale, (255, 255, 255), thickness)
                self.intro_played = False # Reset intro for next user
            
            elif self.state == "HOME":
                if not self.intro_played:
                    speak("Welcome to Wayanad Smart Tourism. I am your virtual guide. Please select a destination to explore.")
                    self.intro_played = True
                self.draw_home(frame)

            elif self.state == "INFO":
                self.draw_info(frame)

            
            # HAND CURSOR
            if result.hand_landmarks:
                self.last_active = time.time()
                hand = result.hand_landmarks[0]
                self.cursor_pos = (int(hand[8].x * WIDTH), int(hand[8].y * HEIGHT)) # Landmark 8 = index finger tip
                
                # Draw Modern Cursor
                cv2.circle(frame, self.cursor_pos, 10, (255, 255, 255), -1) # Draw cursor center
                cv2.circle(frame, self.cursor_pos, 20, (255, 255, 255), 2) # Draw outer cursor ring

                if self.state == "IDLE": self.state = "HOME"
                elif self.state == "HOME": self.check_selection(self.cursor_pos[0], self.cursor_pos[1], hand)
                elif self.state == "INFO":
                    if np.linalg.norm(np.array(self.cursor_pos) - np.array([1200, 80])) < 40:
                        self.state = "HOME"
                
                
            # Show window
            cv2.imshow("Wayanad Smart Kiosk", frame)
            if cv2.waitKey(1) & 0xFF == 27: break
        
        cap.release()
        cv2.destroyAllWindows()

    def draw_home(self, frame):
        for i, name in enumerate(ATTRACTIONS.keys()):
            col, row = i % 3, i // 3
            x, y = 100 + (col * 380), 100 + (row * 300)
            frame[y:y+200, x:x+300] = cv2.resize(IMAGES[name][0], (300, 200))
            cv2.putText(frame, name, (x+10, y+230), 0, 0.7, (255, 255, 255), 2)
            # Weather Overlay on Card
            cv2.putText(frame, get_weather("Wayanad"), (x+10, y+30), 0, 0.5, (255, 255, 255), 1)

    def draw_info(self, frame):
        data = ATTRACTIONS[self.selected]
        cv2.putText(frame, self.selected, (50, 80), 0, 1.5, (0, 0, 0), 3)
        
        # Slideshow
        gallery = IMAGES[self.selected]
        if time.time() - self.last_img_change > 3.0:
            self.img_index = (self.img_index + 1) % len(gallery)
            self.last_img_change = time.time()
        frame[150:400, 50:450] = gallery[self.img_index]

        # Details
        cv2.rectangle(frame, (500, 150), (1200, 420), (30,30,30), -1)
        cv2.putText(frame, f"Activities: {data['act']}", (520, 200), 0, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Timings: {data['time']}", (520, 270), 0, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Entry: {data['fee']}", (520, 340), 0, 0.8, (0, 200, 255), 2)

        # QR
        frame[460:610, 520:670] = QR_CODES[self.selected]
        cv2.putText(frame, "SCAN FOR MAP", (520, 630), 0, 0.5, (255, 255, 255), 1)
        
        # Back Circle
        cv2.circle(frame, (1200, 80), 40, (0,0,0), -1)
        cv2.putText(frame, "BACK", (1175, 85), 0, 0.5, (255,255,255), 2)

    def check_selection(self, px, py, hand):
        for i, name in enumerate(ATTRACTIONS.keys()):
            col, row = i % 3, i // 3
            tx, ty = 100 + (col * 380), 100 + (row * 300)
            if tx < px < tx + 300 and ty < py < ty + 200:
                # 2-finger select
                if sum(1 for t in [8, 12, 16, 20] if hand[t].y < hand[t-2].y) == 2:
                    self.selected, self.state, self.img_index = name, "INFO", 0
                    speak(f"Opening {name}. {ATTRACTIONS[name]['desc']}")

if __name__ == "__main__":
    WayanadUltimateKiosk().run()