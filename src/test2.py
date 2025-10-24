import sounddevice as sd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

   # Configuration for audio capture
samplerate = 44100  # samples per second
duration = 5      # seconds of audio to capture
channels = 1        # mono audio

print(f"Recording for {duration} seconds...")

# Capture audio from the default input device
# You can specify a device ID if needed, e.g., sd.rec(..., device=1)
audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate,channels=channels, dtype='float32')
sd.wait() # Wait until recording is finished

print("Recording finished. Processing audio...")

# Compute the Short-Time Fourier Transform (STFT)
# This generates the spectrogram data
stft = librosa.stft(audio_data[:, 0]) # Use the first channel if stereo
   
# Convert the complex spectrogram to magnitude (amplitude) spectrogram
stft_magnitude = np.abs(stft)

# Convert amplitude spectrogram to decibels for visualization
D_db = librosa.amplitude_to_db(stft_magnitude, ref=np.max)

# Plot the spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(D_db, sr=samplerate, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Dynamic Spectrum (Spectrogram)')
plt.tight_layout()
plt.show()
