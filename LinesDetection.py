from picamera2 import Picamera2
import cv2
import numpy as np

# Inicializar Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start()

# Función para detectar y dibujar líneas
def detect_and_draw_hough_lines(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)  # Detectar bordes
    
    # Detectar líneas con HoughLines
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=150)
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Dibujar línea en rojo
    return frame

# Procesar fotogramas en tiempo real
try:
    while True:
        frame = picam2.capture_array()  # Capturar el fotograma
        frame_with_lines = detect_and_draw_hough_lines(frame)  # Detectar y dibujar líneas
        cv2.imshow("Hough Lines - Live Feed", frame_with_lines)  # Mostrar resultado

        # Salir al presionar 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    picam2.stop()
    cv2.destroyAllWindows()
