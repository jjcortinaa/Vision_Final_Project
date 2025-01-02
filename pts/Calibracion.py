import cv2
import numpy as np
from picamera import PiCamera
from picamera.array import PiRGBArray
import time

# --- Parámetros del tablero de ajedrez ---
chessboard_size = (9, 6)  # Número de esquinas interiores (filas, columnas)
square_size = 1.0  # Tamaño de cada cuadrado en unidades reales (por ejemplo, 1 cm)

# --- Configurar la cámara ---
camera = PiCamera()
camera.resolution = (640, 480)
raw_capture = PiRGBArray(camera, size=(640, 480))
time.sleep(2)  # Esperar a que la cámara esté lista

try:
    # --- Capturar la imagen ---
    print("Capturando imagen del tablero de ajedrez...")
    camera.capture(raw_capture, format="bgr")
    image = raw_capture.array
    cv2.imshow("Tablero capturado", image)
    cv2.imwrite("chessboard.jpg", image)
    print("Imagen guardada como 'chessboard.jpg'.")
