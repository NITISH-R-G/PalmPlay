"""
PalmPlay - Gesture-Controlled Music Player
A beautiful Streamlit-based music player with hand gesture controls.
"""

import streamlit as st
import cv2
import numpy as np
import os
import time
import base64
from io import BytesIO
from PIL import Image

# Import gesture detection and music control
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pygame
from collections import deque

# Try to import mutagen for cover art extraction
try:
    from mutagen.mp3 import MP3
    from mutagen.id3 import ID3, APIC

    HAS_MUTAGEN = True
except ImportError:
    HAS_MUTAGEN = False

# Page configuration
st.set_page_config(
    page_title="PalmPlay",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium dark theme
st.markdown(
    """
<style>
    /* Dark theme base */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f0f23 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(20, 20, 35, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Glass morphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        margin: 10px 0;
    }
    
    /* Now playing card */
    .now-playing {
        background: linear-gradient(135deg, rgba(100, 200, 100, 0.2) 0%, rgba(50, 150, 50, 0.1) 100%);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(100, 200, 100, 0.3);
    }
    
    /* Cover art container */
    .cover-art {
        width: 200px;
        height: 200px;
        border-radius: 12px;
        overflow: hidden;
        margin: 0 auto 15px auto;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
    }
    
    /* Track list item */
    .track-item {
        padding: 12px 15px;
        margin: 5px 0;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .track-item:hover {
        background: rgba(255, 255, 255, 0.1);
    }
    
    .track-item.active {
        background: linear-gradient(90deg, rgba(100, 200, 100, 0.3) 0%, rgba(100, 200, 100, 0.1) 100%);
        border-left: 3px solid #64c864;
    }
    
    /* Control buttons */
    .control-btn {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.1);
        border: none;
        color: white;
        font-size: 24px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .control-btn:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: scale(1.1);
    }
    
    .control-btn.play {
        width: 70px;
        height: 70px;
        background: linear-gradient(135deg, #64c864 0%, #4aa84a 100%);
    }
    
    /* Progress bar */
    .progress-container {
        width: 100%;
        height: 6px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 3px;
        margin: 15px 0;
        overflow: hidden;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #64c864 0%, #8ce68c 100%);
        border-radius: 3px;
        transition: width 0.3s ease;
    }
    
    /* Gesture indicator */
    .gesture-indicator {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: rgba(0, 0, 0, 0.8);
        padding: 30px 50px;
        border-radius: 20px;
        font-size: 24px;
        color: white;
        z-index: 1000;
        animation: fadeInOut 1s ease;
    }
    
    @keyframes fadeInOut {
        0% { opacity: 0; transform: translate(-50%, -50%) scale(0.8); }
        20% { opacity: 1; transform: translate(-50%, -50%) scale(1); }
        80% { opacity: 1; }
        100% { opacity: 0; }
    }
    
    /* Hide streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 4px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ============== Helper Functions ==============


def get_cover_art(file_path):
    """Extract cover art from MP3 file."""
    if not HAS_MUTAGEN:
        return None

    try:
        audio = MP3(file_path, ID3=ID3)
        for tag in audio.tags.values():
            if isinstance(tag, APIC):
                return Image.open(BytesIO(tag.data))
    except:
        pass
    return None


def create_default_cover():
    """Create a default cover art image."""
    img = Image.new("RGB", (200, 200), color=(40, 40, 60))
    return img


def image_to_base64(img):
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


# ============== Session State Initialization ==============

if "music_folder" not in st.session_state:
    st.session_state.music_folder = None
if "tracks" not in st.session_state:
    st.session_state.tracks = []
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False
if "volume" not in st.session_state:
    st.session_state.volume = 50
if "shuffle" not in st.session_state:
    st.session_state.shuffle = False
if "repeat" not in st.session_state:
    st.session_state.repeat = False
if "pygame_init" not in st.session_state:
    pygame.mixer.init()
    st.session_state.pygame_init = True
if "last_gesture" not in st.session_state:
    st.session_state.last_gesture = None
if "gesture_time" not in st.session_state:
    st.session_state.gesture_time = 0


def load_tracks(folder):
    """Load all audio tracks from folder."""
    tracks = []
    if os.path.isdir(folder):
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith((".mp3", ".wav", ".ogg")):
                tracks.append(
                    {
                        "name": os.path.splitext(f)[0],
                        "path": os.path.join(folder, f),
                        "filename": f,
                    }
                )
    return tracks


def play_track(idx):
    """Play a specific track."""
    if 0 <= idx < len(st.session_state.tracks):
        st.session_state.current_idx = idx
        track = st.session_state.tracks[idx]
        pygame.mixer.music.load(track["path"])
        pygame.mixer.music.play()
        st.session_state.is_playing = True


def toggle_play():
    """Toggle play/pause."""
    if st.session_state.is_playing:
        pygame.mixer.music.pause()
        st.session_state.is_playing = False
    else:
        pygame.mixer.music.unpause()
        st.session_state.is_playing = True


def next_track():
    """Play next track."""
    if st.session_state.shuffle:
        import random

        idx = random.randint(0, len(st.session_state.tracks) - 1)
    else:
        idx = (st.session_state.current_idx + 1) % len(st.session_state.tracks)
    play_track(idx)


def prev_track():
    """Play previous track."""
    idx = (st.session_state.current_idx - 1) % len(st.session_state.tracks)
    play_track(idx)


def set_volume(vol):
    """Set playback volume."""
    st.session_state.volume = vol
    pygame.mixer.music.set_volume(vol / 100)


# ============== Main App ==============


def main():
    # Header
    st.markdown(
        """
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="font-size: 2.5em; margin: 0;">
            🎵 <span style="background: linear-gradient(90deg, #64c864, #8ce68c); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">PalmPlay</span>
        </h1>
        <p style="color: rgba(255,255,255,0.6); margin-top: 5px;">Control your music with hand gestures</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar - Playlist
    with st.sidebar:
        st.markdown("### 📂 Music Library")

        # Folder selection
        folder = st.text_input(
            "Music Folder Path:", value=st.session_state.music_folder or ""
        )
        if st.button("📁 Load Folder", use_container_width=True):
            if os.path.isdir(folder):
                st.session_state.music_folder = folder
                st.session_state.tracks = load_tracks(folder)
                st.success(f"Loaded {len(st.session_state.tracks)} tracks!")
            else:
                st.error("Invalid folder path")

        st.markdown("---")

        # Track list
        if st.session_state.tracks:
            st.markdown("### 🎶 Playlist")
            for i, track in enumerate(st.session_state.tracks):
                is_current = i == st.session_state.current_idx
                icon = "▶️" if is_current and st.session_state.is_playing else "🎵"

                col1, col2 = st.columns([1, 6])
                with col1:
                    st.write(icon)
                with col2:
                    if st.button(
                        track["name"][:30], key=f"track_{i}", use_container_width=True
                    ):
                        play_track(i)
        else:
            st.info("Select a music folder to get started")

    # Main content
    col_camera, col_player = st.columns([2, 1])

    with col_camera:
        st.markdown("### 📹 Gesture Control")
        st.markdown(
            """
        <div class="glass-card">
            <p style="text-align: center; color: rgba(255,255,255,0.7);">
                Use these gestures to control playback:
            </p>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-top: 15px;">
                <div style="text-align: center;">
                    <div style="font-size: 2em;">✊</div>
                    <small>Play/Pause</small>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2em;">👉</div>
                    <small>Next</small>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2em;">👈</div>
                    <small>Previous</small>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2em;">✌️</div>
                    <small>Volume</small>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2em;">✋</div>
                    <small>Shuffle</small>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2em;">👍</div>
                    <small>Repeat</small>
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Live Camera Feed
        st.markdown("#### 📷 Camera Feed")
        camera_image = st.camera_input(
            "Show your hand gestures here", label_visibility="collapsed"
        )

    with col_player:
        st.markdown("### 🎧 Now Playing")

        if st.session_state.tracks:
            current_track = st.session_state.tracks[st.session_state.current_idx]

            # Cover art
            cover = get_cover_art(current_track["path"])
            if cover is None:
                cover = create_default_cover()

            # Display cover art
            cover_resized = cover.resize((200, 200))
            st.image(cover_resized, use_container_width=True)

            # Track info
            st.markdown(
                f"""
            <div style="text-align: center; margin: 15px 0;">
                <h3 style="margin: 0; color: white;">{current_track['name'][:40]}</h3>
                <p style="color: rgba(255,255,255,0.6); margin: 5px 0;">Track {st.session_state.current_idx + 1} of {len(st.session_state.tracks)}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Volume slider
            volume = st.slider(
                "🔊 Volume", 0, 100, st.session_state.volume, key="vol_slider"
            )
            if volume != st.session_state.volume:
                set_volume(volume)

            # Playback controls
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                if st.button("🔀", help="Shuffle"):
                    st.session_state.shuffle = not st.session_state.shuffle

            with col2:
                if st.button("⏮️", help="Previous"):
                    prev_track()

            with col3:
                play_icon = "⏸️" if st.session_state.is_playing else "▶️"
                if st.button(play_icon, help="Play/Pause"):
                    if st.session_state.tracks:
                        if (
                            not pygame.mixer.music.get_busy()
                            and not st.session_state.is_playing
                        ):
                            play_track(st.session_state.current_idx)
                        else:
                            toggle_play()

            with col4:
                if st.button("⏭️", help="Next"):
                    next_track()

            with col5:
                if st.button("🔁", help="Repeat"):
                    st.session_state.repeat = not st.session_state.repeat

            # Status indicators
            status_text = []
            if st.session_state.shuffle:
                status_text.append("🔀 Shuffle ON")
            if st.session_state.repeat:
                status_text.append("🔁 Repeat ON")

            if status_text:
                st.markdown(
                    f"<p style='text-align: center; color: #64c864; font-size: 0.9em;'>{' | '.join(status_text)}</p>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                """
            <div style="text-align: center; padding: 40px; color: rgba(255,255,255,0.5);">
                <div style="font-size: 4em; margin-bottom: 15px;">🎵</div>
                <p>No music loaded</p>
                <p style="font-size: 0.9em;">Select a folder from the sidebar</p>
            </div>
            """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
