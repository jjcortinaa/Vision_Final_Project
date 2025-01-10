# Vision_Final_Project
Sistema de Visión por Ordenador para Seguimiento de Pelota y Conteo de Goles
Este proyecto implementa un sistema de visión por ordenador que utiliza una cámara para detectar, seguir una pelota y contabilizar goles al cruzar la línea de una portería. Además, incluye un módulo de autenticación basado en patrones geométricos.

Funcionalidades principales
Calibracion de camara:

Encuentra los parametros de la camara al analizar las fotos hechas al tablero de ajedrez devuelve el error y parametros intrinsecos. 

Sistema de autenticación:

Detecta figuras geométricas (rombo, triángulo, rectángulo y cuadrado) en una secuencia específica para desbloquear el sistema.
Reinicia la secuencia si se detecta un patrón incorrecto.
Seguimiento de pelota y conteo de goles:

Detección de la pelota mediante una máscara de color naranja.
Estimación de la distancia usando el parámetro focal y el tamaño de la pelota.
Registro de goles solo cuando la pelota cruza la portería, evitando duplicados.
Cuadro visual que indica si la pelota está en nuestro campo (rojo) o en el contrario (verde).
Visualización en tiempo real:

Indicador de distancia en pantalla.
Contador de goles dinámico.

Requisitos:
Hardware: Raspberry Pi con cámara o dispositivo equivalente.
Software: Python 3.x y librerías:
OpenCV