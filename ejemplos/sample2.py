import sounddevice as sd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

# --- Configuration ---
SR = 22050  # Sample rate (Hz)
DURATION = 0.1 # Duration of each audio chunk to process (seconds)
WINDOW_SIZE = 2048 # FFT window size for STFT
HOP_LENGTH = 512 # Hop length for STFT
BUFFER_SIZE = 10 # Number of spectrogram frames to display for "dynamic" effect

# --- Setup for Plotting ---
fig, ax = plt.subplots(figsize=(10, 4))
img = None # Placeholder for the spectrogram image

# --- Audio Callback Function ---
def audio_callback(indata, frames, time, status):
    global img
    if status:
        print(status, flush=True)

    y = indata[:, 0].flatten() # Get mono audio data

    # Compute STFT and convert to decibels
    D = librosa.stft(y, n_fft=WINDOW_SIZE, hop_length=HOP_LENGTH)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Store recent spectrogram frames in a deque
    if not hasattr(audio_callback, 'spectrogram_buffer'):
        audio_callback.spectrogram_buffer = deque(maxlen=BUFFER_SIZE)
    audio_callback.spectrogram_buffer.append(S_db)

    # Concatenate the buffer for display
    current_spectrogram = np.hstack(audio_callback.spectrogram_buffer)

    # Update the plot
    if img is None:
        img = librosa.display.specshow(current_spectrogram, sr=SR, hop_length=HOP_LENGTH,
                                       x_axis='time', y_axis='log', ax=ax)
        fig.colorbar(img, format='%+2.0f dB', ax=ax)
        ax.set(title='Dynamic Spectrogram of Microphone Input')
    else:
        # Update the image data and redraw
        img.set_array(current_spectrogram)
        ax.autoscale_view()
        fig.canvas.draw_idle()
    plt.pause(0.001) # Small pause to allow plot to update

# --- Main Execution ---
if __name__ == "__main__":
    try:
        with sd.InputStream(samplerate=SR, blocksize=int(SR * DURATION),
                            channels=1, callback=audio_callback):
            print("Recording... Press Ctrl+C to stop.")
            plt.show() # Display the plot and keep it open
    except KeyboardInterrupt:
        print("\nRecording stopped.")
    except Exception as e:
        print(f"An error occurred: {e}")

