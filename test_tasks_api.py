try:
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import HandLandmarkerOptions, HandLandmarker
    from mediapipe.tasks.python.vision import RunningMode

    print("MediaPipe Tasks API is available.")
except ImportError as e:
    print(f"MediaPipe Tasks API missing: {e}")
