import cv2
from picamera2 import Picamera2

def stream_video():
    # Inicializar Picamera2
    picam = Picamera2()
    picam.preview_configuration.main.size = (1280, 720)  # Configurar la resolución
    picam.preview_configuration.main.format = "RGB888"  # Configurar formato de color
    picam.preview_configuration.align()  # Alineación de configuración
    picam.configure("preview")  # Configurar para vista previa
    picam.start()  # Iniciar la cámara

    # Configurar el VideoWriter para guardar el video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Formato de compresión (XVID)
    out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (1280, 720))  # Guardar en 'output_video.avi'

    while True:
        frame = picam.capture_array()  # Capturar un fotograma
        out.write(frame)  # Escribir el fotograma en el archivo de video

        cv2.imshow("picam", frame)  # Mostrar el video en pantalla

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Salir cuando se presiona 'q'
            break

    # Liberar el objeto VideoWriter y cerrar ventanas
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    stream_video()
