import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# --- Parámetros ---
frame_rate = 30  # Hz
sample_rate = 44100 # Tasa de muestreo del audio
fft_size = 2048
hop_length = 512 # Salto entre ventanas de FFT

# --- Inicialización de la figura y los ejes ---
fig, ax = plt.subplots()
img = ax.imshow(np.zeros((fft_size // 2, 100)), aspect='auto', origin='lower',
                extent=[0, 10, 0, 10]) # Inicializa una imagen vacía
ax.set_xlabel("Tiempo (s)")
ax.set_ylabel("Frecuencia (Hz)")
ax.set_title("Espectrograma en tiempo real")

# --- Función de actualización ---
def update(frame):
    # Obtener un bloque de datos de audio (ejemplo con datos de entrada de audio)
    # En un caso real, aquí leerías un fragmento de audio del micrófono.
    # Usaremos un array de ejemplo que se actualizará en cada frame.
    new_data = np.random.randn(hop_length)
    
    # Calcular el espectrograma del nuevo bloque
    frequencies, times, spectrogram = signal.spectrogram(new_data, fs=sample_rate,
                                                         nperseg=fft_size,
                                                         noverlap=fft_size - hop_length)
    
    # Actualizar la imagen del espectrograma
    # `img.set_data()` es la forma de actualizar los datos de la imagen existente
    # `img.set_data()` toma los datos espectrogramas de la última ventana.
    img.set_data(spectrogram)

    # Mover el eje X para dar la impresión de movimiento
    # El eje X se actualiza dinámicamente a medida que llegan nuevos datos.
    # Esta línea puede necesitar ajustes dependiendo de cómo quieras mover el eje X.
    # Ejemplo: `ax.set_xlim()` para desplazar la ventana de visualización.
    
    # La función `update` se llama 30 veces por segundo.
    return img,

# --- Creación de la animación ---
# La animación llama a la función 'update' 30 veces por segundo.
ani = animation.FuncAnimation(fig, update, frames=None,
                              interval=1000/frame_rate, blit=True,
                              cache_frame_data=False)

plt.show()

