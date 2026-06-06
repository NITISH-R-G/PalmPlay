import numpy as np
import scipy.io.wavfile as wav
import os


def generate_sine_wave(frequency, duration, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = 0.5 * np.sin(2 * np.pi * frequency * t)
    return (wave * 32767).astype(np.int16)


fs = 44100
# Generate a simple melody
notes = [440, 554, 659, 880]  # A C# E A
audio = np.array([], dtype=np.int16)
for note in notes:
    audio = np.concatenate((audio, generate_sine_wave(note, 0.5, fs)))

if not os.path.exists("local_music"):
    os.makedirs("local_music")

wav.write("local_music/test_audio.wav", fs, audio)
print("Created local_music/test_audio.wav")
