"""
PalmPlay - FastAPI Backend Server
Handles music playback, gesture detection, and serves the web frontend.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import base64
import asyncio
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import pygame
from collections import deque
import time

# MediaPipe imports
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Mutagen for cover art
try:
    from mutagen.mp3 import MP3
    from mutagen.id3 import ID3, APIC

    HAS_MUTAGEN = True
except ImportError:
    HAS_MUTAGEN = False

app = FastAPI(title="PalmPlay API")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global state
class MusicPlayer:
    def __init__(self):
        pygame.mixer.init()
        self.tracks = []
        self.current_idx = 0
        self.is_playing = False
        self.volume = 50
        self.shuffle = False
        self.repeat = False
        self.music_folder = None

    def load_folder(self, folder_path):
        self.music_folder = folder_path
        self.tracks = []
        if os.path.isdir(folder_path):
            for f in sorted(os.listdir(folder_path)):
                if f.lower().endswith((".mp3", ".wav", ".ogg")):
                    self.tracks.append(
                        {
                            "name": os.path.splitext(f)[0],
                            "path": os.path.join(folder_path, f),
                            "filename": f,
                        }
                    )
        return len(self.tracks)

    def play_track(self, idx):
        if 0 <= idx < len(self.tracks):
            self.current_idx = idx
            pygame.mixer.music.load(self.tracks[idx]["path"])
            pygame.mixer.music.play()
            self.is_playing = True
            return True
        return False

    def toggle_play(self):
        if self.is_playing:
            pygame.mixer.music.pause()
            self.is_playing = False
        else:
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.unpause()
            else:
                if self.tracks:
                    pygame.mixer.music.play()
            self.is_playing = True
        return self.is_playing

    def next_track(self):
        if self.tracks:
            if self.shuffle:
                import random

                self.current_idx = random.randint(0, len(self.tracks) - 1)
            else:
                self.current_idx = (self.current_idx + 1) % len(self.tracks)
            return self.play_track(self.current_idx)
        return False

    def prev_track(self):
        if self.tracks:
            self.current_idx = (self.current_idx - 1) % len(self.tracks)
            return self.play_track(self.current_idx)
        return False

    def set_volume(self, vol):
        self.volume = max(0, min(100, vol))
        pygame.mixer.music.set_volume(self.volume / 100)
        return self.volume

    def get_state(self):
        current_track = None
        if self.tracks and 0 <= self.current_idx < len(self.tracks):
            current_track = self.tracks[self.current_idx]
        return {
            "tracks": [
                {"name": t["name"], "filename": t["filename"]} for t in self.tracks
            ],
            "current_idx": self.current_idx,
            "current_track": current_track["name"] if current_track else None,
            "is_playing": self.is_playing,
            "volume": self.volume,
            "shuffle": self.shuffle,
            "repeat": self.repeat,
        }


# Gesture Recognizer
class GestureRecognizer:
    def __init__(self):
        self.center_buf = deque(maxlen=6)
        self.finger_buf = deque(maxlen=6)
        self.last_trigger = {}
        self.cooldown = 1.0

    def fingers_up(self, lm):
        tips = [8, 12, 16, 20]
        count = 0
        for tip in tips:
            try:
                if lm[tip][1] < lm[tip - 2][1]:
                    count += 1
            except:
                pass
        try:
            if abs(lm[4][0] - lm[0][0]) > 0.06:
                count += 1
        except:
            pass
        return count

    def cooldown_ok(self, action):
        t = time.time()
        last = self.last_trigger.get(action, 0)
        if t - last >= self.cooldown:
            self.last_trigger[action] = t
            return True
        return False

    def recognize(self, landmarks):
        if not landmarks:
            return None

        lm = landmarks
        raw_cnt = self.fingers_up(lm)
        self.finger_buf.append(raw_cnt)
        cnt = int(round(np.median(list(self.finger_buf))))

        # Check finger states
        thumb_up = lm[4][1] < lm[3][1]
        idx_up = lm[8][1] < lm[6][1]
        mid_up = lm[12][1] < lm[10][1]
        ring_up = lm[16][1] < lm[14][1]
        pinky_up = lm[20][1] < lm[18][1]

        # Open palm -> Shuffle
        if cnt >= 5 and idx_up and mid_up and ring_up and pinky_up and thumb_up:
            if self.cooldown_ok("shuffle"):
                return "shuffle"

        # Thumb up -> Repeat
        if thumb_up and not idx_up and not mid_up and not ring_up and not pinky_up:
            if lm[4][1] < lm[0][1] - 0.1:
                if self.cooldown_ok("repeat"):
                    return "repeat"

        # Two fingers -> Volume (return special)
        if cnt >= 2 and idx_up and mid_up and not ring_up and not pinky_up:
            vol = int((1.0 - np.mean([lm[8][1], lm[12][1]])) * 100)
            return ("volume", max(0, min(100, vol)))

        # Fist
        tips = [4, 8, 12, 16, 20]
        folded = sum(1 for tip in tips if lm[tip][1] > lm[tip - 2][1])
        if folded >= 4:
            if self.cooldown_ok("toggle"):
                return "toggle"

        return None


# Initialize global objects
player = MusicPlayer()
gesture_recognizer = GestureRecognizer()

# Hand detector setup
hand_detector = None


def init_hand_detector():
    global hand_detector
    model_path = os.path.abspath("hand_landmarker.task")
    if os.path.exists(model_path):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        hand_detector = vision.HandLandmarker.create_from_options(options)


try:
    init_hand_detector()
except Exception as e:
    print(f"Hand detector init failed: {e}")


# API Routes
@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/api/state")
async def get_state():
    return player.get_state()


@app.post("/api/load-folder")
async def load_folder(data: dict):
    folder = data.get("folder", "")
    if os.path.isdir(folder):
        count = player.load_folder(folder)
        return {"success": True, "count": count}
    return {"success": False, "error": "Invalid folder"}


@app.post("/api/play/{idx}")
async def play_track(idx: int):
    success = player.play_track(idx)
    return {"success": success, "state": player.get_state()}


@app.post("/api/toggle")
async def toggle_play():
    is_playing = player.toggle_play()
    return {"is_playing": is_playing}


@app.post("/api/next")
async def next_track():
    success = player.next_track()
    return {"success": success, "state": player.get_state()}


@app.post("/api/prev")
async def prev_track():
    success = player.prev_track()
    return {"success": success, "state": player.get_state()}


@app.post("/api/volume/{vol}")
async def set_volume(vol: int):
    new_vol = player.set_volume(vol)
    return {"volume": new_vol}


@app.post("/api/shuffle")
async def toggle_shuffle():
    player.shuffle = not player.shuffle
    return {"shuffle": player.shuffle}


@app.post("/api/repeat")
async def toggle_repeat():
    player.repeat = not player.repeat
    return {"repeat": player.repeat}


@app.get("/api/cover/{idx}")
async def get_cover(idx: int):
    if not HAS_MUTAGEN or idx >= len(player.tracks):
        return {"cover": None}

    try:
        audio = MP3(player.tracks[idx]["path"], ID3=ID3)
        for tag in audio.tags.values():
            if isinstance(tag, APIC):
                cover_b64 = base64.b64encode(tag.data).decode()
                return {"cover": f"data:image/jpeg;base64,{cover_b64}"}
    except:
        pass
    return {"cover": None}


@app.post("/api/detect-gesture")
async def detect_gesture(file: UploadFile = File(...)):
    """Process an image frame and detect gestures."""
    if hand_detector is None:
        return {"gesture": None, "error": "Hand detector not initialized"}

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"gesture": None}

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = hand_detector.detect(mp_image)

        if result.hand_landmarks:
            landmarks = [(lm.x, lm.y, lm.z) for lm in result.hand_landmarks[0]]
            gesture = gesture_recognizer.recognize(landmarks)

            if gesture:
                # Execute gesture action
                if gesture == "toggle":
                    player.toggle_play()
                    return {"gesture": "toggle", "action": "play/pause"}
                elif gesture == "shuffle":
                    player.shuffle = not player.shuffle
                    return {"gesture": "shuffle", "value": player.shuffle}
                elif gesture == "repeat":
                    player.repeat = not player.repeat
                    return {"gesture": "repeat", "value": player.repeat}
                elif isinstance(gesture, tuple) and gesture[0] == "volume":
                    player.set_volume(gesture[1])
                    return {"gesture": "volume", "value": gesture[1]}

        return {"gesture": None}
    except Exception as e:
        return {"gesture": None, "error": str(e)}


# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
