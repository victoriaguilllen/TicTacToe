import cv2
import os
from datetime import datetime
from picamera2 import Picamera2

script_dir = os.path.dirname(os.path.abspath(__file__))

def capture_photos():
    # Crear una carpeta para guardar las fotos
    save_path = f"{script_dir}/calibration_photos"
    os.makedirs(save_path, exist_ok=True)
    
    # Inicializar la cámara
    picam = Picamera2()
    picam.preview_configuration.main.size=(640, 480)
    picam.preview_configuration.main.format="RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()
    
    print("Presiona espacio para capturar una foto. Pulsa 'q' para salir.")
    
    photo_count = 0
    max_photos = 10
    
    while photo_count < max_photos:
    
        frame = picam.capture_array()
        cv2.imshow("picam", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key != 255:  # Se presionó una tecla
            if key == ord('q'):  # Salir al presionar 'q'
                break
            elif key == ord(" "):
                # Guardar la imagen con un nombre único basado en la fecha y hora
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{save_path}/photo_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                photo_count += 1
                print(f"Foto guardada: {filename}")
    

    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_photos()
