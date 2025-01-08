import cv2
import numpy as np
import os
from typing import List
from utils import non_max_suppression  # Asegúrate de que utils esté en tu proyecto

cont_triangle, prev_triangle, val_triangle = 0, 0, 0
cont_square, prev_square,val_square = 0, 0, 0
cont_rhombus, prev_rhombus,val_rhombus = 0, 0, 0
cont_rectangle, prev_rectangle,val_rectangle = 0,0, 0
key_one, key_two, key_three, key_four = 0, 0, 0, 0
finished = False
sufficient_cond = 10
dict_keys = {"rhombus": 0, "triang": 1, "rectangle": 2, "square": 3}

current_key = 0

def show_image(img: np.array, img_name: str = "Image"):
    cv2.imshow(img_name, img) 
    cv2.waitKey(1) 
    cv2.destroyAllWindows()

def gaussian_blur(img: np.array, sigma: float, filter_shape: List | None = None, verbose: bool = False) -> np.array:
    if filter_shape is None:
        filter_size = int(8 * sigma + 1)
        filter_shape = [filter_size, filter_size]

    # Crear un filtro Gaussiano
    ax = np.linspace(-(filter_shape[0] // 2), filter_shape[0] // 2, filter_shape[0])
    gauss_filter = np.exp(-0.5 * (ax**2 + ax[:, None]**2) / sigma**2)
    gauss_filter /= np.sum(gauss_filter)  # Normaliza el filtro

    # Aplicar el filtro Gaussiano a la imagen
    gb_img = cv2.filter2D(img, -1, gauss_filter)
    
    if verbose:
        show_image(img=gb_img, img_name=f"Gaussian Blur: Sigma = {sigma}")
    
    return gauss_filter, gb_img.astype(np.uint8)

def sobel_edge_detector(img: np.array, filter: np.array, gauss_sigma: float, gauss_filter_shape: List | None = None, verbose: bool = False) -> np.array:
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, blurred = gaussian_blur(gray_img, gauss_sigma, gauss_filter_shape, verbose)
    
    blurred = blurred / 255.0  
    v_edges = cv2.filter2D(blurred, -1, filter)
    h_edges = cv2.filter2D(blurred, -1, filter.T)
    
    sobel_edges_img = np.hypot(v_edges, h_edges)
    sobel_edges_img = np.clip(sobel_edges_img, 0, 1)

    theta = np.arctan2(h_edges, v_edges)
    
    if verbose:
        show_image(img=(sobel_edges_img * 255).astype(np.uint8), img_name="Sobel Edges")
    
    return np.squeeze(sobel_edges_img), np.squeeze(theta)

def canny_edge_detector(img: np.array, sobel_filter: np.array, gauss_sigma: float, gauss_filter_shape: List | None = None, verbose: bool = False):
    sobel_edges_img, theta = sobel_edge_detector(img, sobel_filter, gauss_sigma, gauss_filter_shape, verbose)
    canny_edges_img = non_max_suppression(sobel_edges_img, theta)
    
    if verbose:
        show_image((canny_edges_img * 255).astype(np.uint8), img_name="Canny Edges")
        
    return canny_edges_img

def classify_shape(approx):
    if check_triangle(approx):
        return "Triangle"
    if check_square(approx):
        return "Square"
    if check_rhombus(approx):
        return "Rhombus"
    if check_rectangle(approx):
        return "Rectangle"
    else:
        return "Unknown"

def check_triangle(approx):
    num_vertices = len(approx)
    if num_vertices==3:
        return True
    else:
        return False

def check_square(approx):
    num_vertices = len(approx)
    if num_vertices==4:
        x, y, w, h = cv2.boundingRect(approx)
        ratio = float(w)/h
        if 0.9<ratio<1.1:
            return True
        else:
            return False
    else:
        return False

def check_rhombus(approx, tolerance: float = 0.2):
    num_vertices = len(approx)
    if num_vertices == 4:  
        side_lengths = [
            np.linalg.norm(approx[i][0] - approx[(i + 1) % 4][0])  # Distancia entre vértices consecutivos
            for i in range(4)
        ]
        
        diag1 = np.linalg.norm(approx[0][0] - approx[2][0])
        diag2 = np.linalg.norm(approx[1][0] - approx[3][0])
        
        # Verificar si los lados son aproximadamente iguales
        mean_length = np.mean(side_lengths)
        if all((1 - tolerance) * mean_length <= length <= (1 + tolerance) * mean_length for length in side_lengths):
            # Verificar si las diagonales se cruzan en un ángulo significativo
            diag_ratio = diag1 / diag2 if diag2 > 0 else 0
            if 0.4 <= diag_ratio <= 2.5:  # Las diagonales de un rombo no pueden ser extremadamente desiguales
                return True
    return False

def check_rectangle(approx):
    num_vertices = len(approx)
    if num_vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        ratio = float(w) / h
        if 0.1 < ratio < 0.7 or 1.5<ratio<3:  # Los rectángulos tienen proporciones más amplias
            return True
    return False

def detect_shapes(img: np.array, canny_sigma: float, sobel_filter: np.array, min_area: int = 1000):
    global cont_triangle, cont_square, cont_rhombus, cont_rectangle
    canny_edges = canny_edge_detector(img, sobel_filter, canny_sigma)
    
    contours, _ = cv2.findContours((canny_edges * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shape_name = "None"
    detected_shapes = img.copy()  # Fondo original de la imagen
    
    for contour in contours:
        # Filtrar contornos pequeños (evita ruido)
        if cv2.contourArea(contour) < min_area:
            continue
        
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        shape_name = classify_shape(approx)
        
        # Calcula el color promedio dentro del contorno
        mask = np.zeros_like(img[:,:,0])  # Máscara para el contorno actual
        cv2.drawContours(mask, [contour], -1, 255, -1)  # Dibuja el contorno en blanco sobre la máscara negra
        mean_color = cv2.mean(img, mask=mask)[:3]  # Promedio del color BGR
        
        # Convierte el color a formato de texto legible
        color_name = f"BGR({int(mean_color[0])}, {int(mean_color[1])}, {int(mean_color[2])})"
        
        # Dibuja la forma detectada en la imagen original
        cv2.drawContours(detected_shapes, [approx], 0, (0, 255, 0), 2)
        
        if shape_name != "Unknown":
            print(f"Detected: {shape_name}, Color: {color_name}")
        if shape_name=="Triangle":
            cont_triangle +=1
        if shape_name=="Square":
            cont_square+=1
        if shape_name=="Rhombus":
            cont_rhombus+=1
            print(cont_rhombus)
        if shape_name=="Rectangle":
            cont_rectangle+=1
            
    
    return detected_shapes

def main_pattern_detection(videopath):
    global finished
    cap = cv2.VideoCapture(videopath)  # Accede a la cámara por defecto

    # Filtro Sobel utilizado para la detección de bordes
    sobel_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    gauss_sigma = 1.0   # Parámetro de sigma para el filtro Gaussiano
    
    while not finished:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo capturar el video.")
            break
        
        # Aplica la detección de formas
        detected_shapes_img = detect_shapes(frame, gauss_sigma, sobel_filter)
        

        # Muestra la imagen resultante con las formas detectadas
        cv2.imshow("Detected Shapes on Black Background", detected_shapes_img)

        validate_square()
        validate_triang()
        validate_rhombus()
        validate_rectangle()

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    

    cap.release()
    cv2.destroyAllWindows()

def validate_square():
    global cont_square,prev_square,val_square, key_three, current_key
    if cont_square==prev_square:
        val_square = 0
    else:
        val_square +=1
        prev_square = cont_square
    
    if val_square==sufficient_cond:
        if current_key not in [dict_keys["square"],dict_keys["square"]+1]:
            current_key = 0
        else:
            key_three +=1
        print("SQUARE DETECTED")

def validate_triang():
    global val_triangle,prev_triangle,cont_triangle, key_two, current_key
    if cont_triangle==prev_triangle:
        val_triangle = 0
        
    else:
        val_triangle +=1
        prev_triangle = cont_triangle
    
    if val_triangle ==sufficient_cond:
        if current_key not in [dict_keys["triang"],dict_keys["triang"]+1]:
            current_key ==0
        else:
            key_two+=1
        print("TRIANGLE DETECTED")

def validate_rhombus():
    global cont_rhombus,prev_rhombus,val_rhombus, key_one, current_key
    if cont_rhombus==prev_rhombus:
        val_rhombus = 0
    else:
        val_rhombus +=1
        prev_rhombus = cont_rhombus
    
    if val_rhombus==sufficient_cond:
        if current_key not in [dict_keys["rhombus"],dict_keys["rhombus"]+1]:
            current_key = 0
        else:
            key_one +=1
        print("RHOMBUS DETECTED")

def validate_rectangle():
    global cont_rectangle, prev_rectangle, val_rectangle, key_four, current_key
    if cont_rectangle == prev_rectangle:
        val_rectangle = 0
    else:
        val_rectangle += 1
        prev_rectangle = cont_rectangle
    
    if val_rectangle == sufficient_cond:
        if current_key not in [dict_keys["rectangle"], dict_keys["rectangle"] + 1]:
            current_key = 0
        else:
            key_four += 1
        print("RECTANGLE DETECTED")

def validate_sequence():
    global key_one, key_two, key_three, key_four, current_key, finished
    if key_one == 1 and current_key == 0:
        current_key += 1
        key_one = 0
    if key_two == 1 and current_key == 1:
        current_key += 1
        key_two = 0
    if key_three == 1 and current_key == 2:
        current_key += 1
        key_three = 0
    if key_four == 1 and current_key == 3:
        current_key += 1
        key_four = 0

    if current_key == 4:  # Ahora la secuencia incluye el rectángulo
        finished = True

if __name__ == "__main__":
    main_pattern_detection("src/4_gf_def.mp4")
