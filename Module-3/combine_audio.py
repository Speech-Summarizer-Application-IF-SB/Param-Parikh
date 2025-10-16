import librosa
import soundfile as sf
import numpy as np

# Load both speaker audio files
audio1, sr1 = librosa.load("audio/sample.wav", sr=16000)
audio2, sr2 = librosa.load("audio/synthetic.wav", sr=16000)

# Ensure both have same sampling rate
if sr1 != sr2:
    raise ValueError("Sampling rates must match!")

# Add a small silence between the clips (for clarity)
silence = np.zeros(int(sr1 * 1.0))  # 1 second of silence

# Concatenate the audios
combined = np.concatenate([audio1, silence, audio2])

# Save the combined audio file
sf.write("audio/combined_meeting.wav", combined, sr1)

print("✅ Combined audio saved as audio/combined_meeting.wav")
