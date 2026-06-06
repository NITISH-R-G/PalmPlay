import mediapipe as mp

try:
    print(f"mp file: {mp.__file__}")
except:
    print("mp has no __file__")

try:
    print(dir(mp))
except:
    print("Cannot dir(mp)")

try:
    import mediapipe.python.solutions as solutions

    print("Imported mediapipe.python.solutions successfully")
except ImportError as e:
    print(f"Failed to import mediapipe.python.solutions: {e}")
