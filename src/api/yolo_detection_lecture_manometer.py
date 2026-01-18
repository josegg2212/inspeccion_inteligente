from ultralytics import YOLO
import cv2

class YoloDetectorLecture:

    def __init__(self, model_path):
        # Carga del modelo YOLO entrenado para lectura
        self.model = YOLO(model_path)
        # Mapeo de clases entrenadas a nombres semanticamente utiles
        self.class_map = {
            0: 'base',
            1: 'maximum',
            2: 'minimum',
            3: 'tip'
        }

    def process_image(self, source, show: bool = False, save: bool = False):
        # Inferencia y extraccion de centroides de cada deteccion
        results = self.model(source=source, show=False, save=False)

        res = results[0]

        points = {}
        # Recorre detecciones y guarda punto central por clase
        for box, cls in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy()):
            x1, y1, x2, y2 = box.astype(int)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            name = self.class_map.get(int(cls), f'class_{int(cls)}')
           
            points[name] = (cx, cy)

        return points

       





