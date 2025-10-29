import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
SAMPLERATE = 44100  # samples per second
#CHUNK = 1024        # number of frames per buffer
CHUNK=44100
CHANNELS = 1        # mono audio
plot_data = np.zeros(CHUNK)
fig, ax = plt.subplots()
line, = ax.plot(plot_data)
ax.set_ylim([-1, 1])  # Adjust y-axis limits according to expected audio amplitude
ax.set_xlim([0, CHUNK])

def audio_callback(indata, frames, time, status):
    global plot_data
    if status:
       print(status)
    plot_data = indata[:, 0]  # Assuming mono input, take the first channel

def update_plot(frame):
    line.set_ydata(plot_data)
    return line,

with sd.InputStream(samplerate=SAMPLERATE, channels=CHANNELS, callback=audio_callback, blocksize=CHUNK):
     ani = FuncAnimation(fig, update_plot, blit=True, interval=50) # interval in milliseconds
     plt.show()
