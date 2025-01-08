import numpy as np
import cv2

dist = 0
prev_dist = 0
focal = 885  # Actualizado para la cámara del iPhone 11 (dependiendo de la distancia focal de la cámara)
pix = 30
width = 6.1  # Ancho promedio del objeto grabado, ajustado según el iPhone 11
kernel = np.ones((3, 3), 'uint8')
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.6
text_color = (255, 255, 255)  # Color del texto (blanco)
rect_color = (0, 0, 0)  # Color del rectángulo (negro)
thickness = 2

goal_distance = 150
midfield_distance = 85
goal_counter = 0

# Cambiar la fuente de entrada de PiCamera a un video
video_path = "src/tracker_def.mp4"  # Ruta al video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: No se pudo abrir el video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fin del video o error en la lectura del frame.")
        break

    # Convertir a HSV
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Límites ajustados para incluir naranjas oscuros y brillantes
    lbound = (10, 100, 100)  # Límite inferior para incluir naranjas más oscuros
    ubound = (25, 255, 255)  # Límite superior para incluir naranjas más brillantes

    # Crear la máscara
    mask = cv2.inRange(hsv_img, lbound, ubound)

    # Aplicar operación morfológica
    d_img = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=5)

    # Encontrar contornos
    cont, hie = cv2.findContours(d_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont = sorted(cont, key=cv2.contourArea, reverse=True)[:1]

    for cnt in cont:
        if 100 < cv2.contourArea(cnt) < 300000:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(np.clip(box, 0, [frame.shape[1] - 1, frame.shape[0] - 1]))

            # Calcular distancia
            pixels = rect[1][0]
            dist = (width * focal) / pixels

            if prev_dist > goal_distance and 95 > dist > 75:
                goal_counter += 1
                print(goal_counter)

            if dist < midfield_distance:
                box_color = (0, 0, 255)
            else:
                box_color = (0, 255, 0)

            cv2.drawContours(frame, [box], -1, box_color, 3)

            # Mostrar distancia
            cv2.putText(frame, str(round(dist, 2)), (110, 50), font, fontScale, text_color, 1, cv2.LINE_AA)

            prev_dist = dist

    # Añadir rectángulo y texto "GOALS: goal_counter"
    text = f"GOALS: {goal_counter}"
    (text_width, text_height), baseline = cv2.getTextSize(text, font, fontScale, thickness)
    top_left = (10, 10)
    bottom_right = (10 + text_width + 10, 10 + text_height + 10)

    # Dibujar rectángulo negro semitransparente
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, rect_color, -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Escribir el texto encima del rectángulo
    cv2.putText(frame, text, (top_left[0] + 5, top_left[1] + text_height + 5), font, fontScale, text_color, thickness, cv2.LINE_AA)

    # Mostrar frame procesado
    cv2.imshow("DISTANCE", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
