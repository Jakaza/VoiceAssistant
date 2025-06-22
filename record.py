import os
import sounddevice as sd
from scipy.io.wavfile import write

SAMPLE_RATE = 16000 # 16kHz
DURATION = 3 # Seconds

def record_commands(label, index):
    print(f"Recording '{label}' sample #{index}...")
    audio = sd.rec(int(SAMPLE_RATE*DURATION), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()

    folder = f"audio/{label}"
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"{label}_{index}.wav")
    write(filename, SAMPLE_RATE, audio)
    print(f"Saved: {filename}")