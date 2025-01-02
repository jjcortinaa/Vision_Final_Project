from picamera2 import Picamera2
import cv2
import numpy as np

# Inicializar Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start()

# Función para detectar y contar las esquinas
def detect_corners(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
    gray = np.float32(gray)  # Convertir a tipo flotante para Harris
    
    # Detección de esquinas usando el algoritmo de Harris
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    
    # Marcar esquinas con un umbral
    dst = cv2.dilate(dst, None)  # Dilatar para mejorar la visualización
    frame[dst > 0.01 * dst.max()] = [0, 0, 255]  # Marcar esquinas en rojo

    # Contar el número de esquinas detectadas
    num_corners = np.sum(dst > 0.01 * dst.max())  # Contar píxeles que cumplen el umbral
    return frame, num_corners

# Procesar fotogramas en tiempo real
try:
    while True:
        frame = picam2.capture_array()  # Capturar el fotograma
        frame_with_corners, num_corners = detect_corners(frame)  # Detectar esquinas

        # Mostrar el número de esquinas detectadas en la pantalla
        cv2.putText(frame_with_corners, f'Esquinas detectadas: {num_corners}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Mostrar el resultado
        cv2.imshow("Detectando Esquinas - Live Feed", frame_with_corners)

        # Salir al presionar 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    picam2.stop()
    cv2.destroyAllWindows()
