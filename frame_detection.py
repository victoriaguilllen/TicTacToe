import cv2
import numpy as np
import time
from picamera2 import Picamera2
from datetime import datetime

# Definir el rango de color a seguir en formato HSV (ajustado para evitar rojos oscuros)
lower_red1 = np.array([0, 180, 100])    # Rango para rojos más claros y brillantes
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 180, 100])  # Rango para rojos más saturados
upper_red2 = np.array([180, 255, 255])

# Configurar Picamera2
picam = Picamera2()
picam.preview_configuration.main.size = (640, 480)  # Ajusta la resolución según sea necesario
picam.preview_configuration.main.format = "RGB888"
picam.preview_configuration.align()
picam.configure("preview")
picam.start()

# Crear una lista para almacenar los puntos del recorrido
trajectory = []

# Crear una imagen en blanco (predefinir tamaño de la imagen si no se detecta rojo)
frame_width = picam.preview_configuration.main.size[0]
frame_height = picam.preview_configuration.main.size[1]
blank_image = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

# Función para calcular la distancia euclidiana entre dos puntos
def euclidean_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

# Umbral de distancia para unir puntos consecutivos
distance_threshold = 50

while True:
    # Captura de la imagen con Picamera2
    frame = picam.capture_array()

    # Convertir la imagen a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.erode(mask, None, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            trajectory.append((cx, cy))
    else:
        pass  # No se detectó rojo en este frame

    # Dibujar la trayectoria en el frame actual
    if len(trajectory) > 1:
        for i in range(1, len(trajectory)):
            pt1 = trajectory[i - 1]
            pt2 = trajectory[i]
            if euclidean_distance(pt1, pt2) <= distance_threshold:
                cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
                cv2.line(blank_image, pt1, pt2, (0, 255, 255), 2)

    # Mostrar el video en tiempo real con la trayectoria dibujada
    cv2.imshow("Trajectory Tracking", frame)

    key = cv2.waitKey(1) & 0xFF  # Capturar la tecla presionada

    if key == ord('s'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"dataset/circles/{timestamp}.jpg", blank_image)
        print(f"Image with trajectory saved as 'dataset/circles/{timestamp}.jpg'")
        break

    elif key == ord('q'):
        break

picam.stop()
cv2.destroyAllWindows()
