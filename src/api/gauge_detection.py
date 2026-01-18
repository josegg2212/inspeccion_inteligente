from ultralytics import YOLO
import cv2

class YoloDetector:
    def __init__(self, modelo_path):
        # Carga del modelo YOLO para deteccion
        self.model = YOLO(modelo_path)
    
    def process_image(self, fuente, show = False, save = False):
        # Ejecuta inferencia sobre la fuente y devuelve imagen anotada + bboxes
        results = self.model(source=fuente, show=show, save=save)

        res = results[0]

        detected_img = res.plot()

        return detected_img, res.boxes.xyxy
       





