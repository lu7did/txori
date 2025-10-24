import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from scipy.signal import spectrogram

# Configuration for audio input
samplerate = 44100  # samples per second
duration = 5      # seconds
channels = 1      # mono

# Record audio
print("Recording audio...")
audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=channels, dtype='float32')
sd.wait()  # Wait until recording is finished
print("Recording finished.")

# Calculate spectrogram
frequencies, times, Sxx = spectrogram(audio_data.flatten(), fs=samplerate)

# Plot the spectrogram
plt.figure(figsize=(10, 6))
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
plt.title("Audio Spectrogram")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Intensity (dB)")
plt.ylim([0, samplerate / 2]) # Limit y-axis to Nyquist frequency
plt.show()
