from __future__ import annotations

import json
import math
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pathlib import Path

from yolo_detection import YoloDetector
from yolo_detection_lecture_manometer import YoloDetectorLecture

# Wrapper basico de FastAPI para definir rutas dinamicamente
class ApiRestWrapper:
    def __init__(self, host: str, port: int, routes=None):
        self.app = FastAPI()
        self.host = host
        self.port = port
        if routes:
            self.setup_routes(routes)

    def setup_routes(self, routes):
        for url, methods, callback in routes:
            self.app.add_api_route(url, callback, methods=methods)

    def run_app(self):
        uvicorn.run(self.app, host=self.host, port=self.port)


# Paths: temp es efimero para cada request; ok es persistente para historico
BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp"
OK_DIR = BASE_DIR / "ok"
OK_RESULTS_DIR = OK_DIR / "results"
OK_INPUTS_DIR = OK_DIR / "inputs"
LAST_OK_PATH = OK_DIR / "last_ok.jpg"

# Model weights (modelos entrenados)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DETECT_MODEL_PATH = str(PROJECT_ROOT / "models" / "manometer_detection_model.pt")
LECTURE_MODEL_PATH = str(PROJECT_ROOT / "models" / "manometer_lecture_model.pt")

# Clase principal del API REST que orquesta el flujo completo
class ManometerDetect:
    """FastAPI endpoint handler for manometer detection and lecture."""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port

        self.MEASURE_UNIT = ""
        self.MEASURE_MAX = 0.0
        self.MEASURE_MIN = 0.0

        # Asegura carpetas persistentes para historico de entradas/salidas
        OK_INPUTS_DIR.mkdir(parents=True, exist_ok=True)
        OK_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        # API wrapper y rutas disponibles
        self.apirest_wrapper = ApiRestWrapper(self.host, self.port)
        self.apirest_wrapper.setup_routes(
            [
                ("/process_camera_image", ["POST"], self.procces_camera_image),
                ("/get_last_ok", ["GET"], self.get_last_ok),
                ("/get_last_ok/{zone_id}", ["GET"], self.get_last_ok_by_zone),
            ]
        )

    def start(self):
        self.apirest_wrapper.run_app()

    # Endpoint principal: recibe imagen + metadata y devuelve imagen anotada
    async def procces_camera_image(self, file: UploadFile = File(...), metadata: UploadFile = File(...)):
        metadata_content = await metadata.read()
        try:
            config_data = json.loads(metadata_content)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid metadata JSON: {e}")

        try:
            # Pipeline completo de procesamiento
            response = self.process_image(file, config_data)

            if response["code"] == 400:
                raise HTTPException(status_code=400, detail=response["message"])

            # Se devuelven mensajes y bboxes en headers para el cliente
            headers = {
                "X-Message": response["message"],
                "X-Bounding-Boxes": json.dumps(response["bboxes"].tolist()),
            }
            return FileResponse(
                response["data"],
                headers=headers,
                media_type="image/jpeg",
                status_code=response["code"],
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    def get_last_ok(self):
        # Endpoint auxiliar para consultar la ultima imagen exitosa
        if not LAST_OK_PATH.exists():
            raise HTTPException(status_code=404, detail="No successful image processed yet")

        return FileResponse(
            str(LAST_OK_PATH),
            media_type="image/jpeg",
            headers={"X-Message": "Last successfully processed image"},
            status_code=200,
        )

    def get_last_ok_by_zone(self, zone_id: str):
        # Endpoint auxiliar para consultar la ultima imagen exitosa por zona
        safe_zone = "".join(c for c in zone_id if c.isalnum() or c in "-_")
        if not safe_zone:
            raise HTTPException(status_code=400, detail="Invalid zone_id")

        zone_path = OK_DIR / f"last_ok_{safe_zone}.jpg"
        if not zone_path.exists():
            raise HTTPException(status_code=404, detail=f"No successful image for zone '{safe_zone}'")

        return FileResponse(
            str(zone_path),
            media_type="image/jpeg",
            headers={"X-Message": f"Last successfully processed image for zone {safe_zone}"},
            status_code=200,
        )

    # Orquestador del flujo: validacion -> temp -> deteccion -> lectura -> salida
    def process_image(self, file: UploadFile, config_data: dict):
        try:
            self.MEASURE_UNIT = config_data["MEASURE_SCALE"]["UNIT"]
            self.MEASURE_MAX = float(config_data["MEASURE_SCALE"]["MAX"])
            self.MEASURE_MIN = float(config_data["MEASURE_SCALE"]["MIN"])
        except Exception as e:
            return {
                "code": 400,
                "message": f"Missing/invalid MEASURE_SCALE in metadata: {e}",
                "data": "",
                "bboxes": np.zeros((0, 4), dtype=int),
            }

        zone_id = config_data.get("ZONE_ID") or config_data.get("zone_id")

        # Limpia temp para esta request
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)
        TEMP_DIR.mkdir(parents=True, exist_ok=True)

        # Guarda el input en disco para reutilizar en el pipeline
        file_path = TEMP_DIR / file.filename
        with open(file_path, "wb") as f:
            f.write(file.file.read())

        # Copia del original para persistir si todo sale bien
        raw_copy_path = TEMP_DIR / f"raw_{file.filename}"
        shutil.copyfile(file_path, raw_copy_path)

        # Deteccion y recorte del manometro
        response = self.detect_manometer(str(file_path))
        if response["code"] != 200:
            return response
        detected_image_path = response["data"]
        bboxes = response["bboxes"]

        # Lectura de angulos con el segundo modelo
        response = self.read_manometer(detected_image_path)
        if response["code"] != 200:
            return response

        angle = response["data"]["final_angle"]
        max_angle = response["data"]["max_angle"]
        min_angle = response["data"]["min_angle"]
        points = response["data"]["points"]

        # Genera imagen anotada final
        out = self.save_output_image(file.filename, detected_image_path, angle, max_angle, min_angle, points, bboxes)

        # Persiste solo si fue una lectura exitosa
        if out.get("code") == 200:
            try:
                self.persist_success(raw_copy_path, Path(out["data"]), file.filename, zone_id=zone_id)
            except Exception:
                pass

        return out

    # Copia entradas/salidas a carpeta persistente y actualiza last_ok
    def persist_success(self, raw_path: Path, result_path: Path, original_filename: str, zone_id: str | None = None) -> None:
        OK_INPUTS_DIR.mkdir(parents=True, exist_ok=True)
        OK_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        # Inputs: keep original filename
        dst_in = OK_INPUTS_DIR / original_filename
        shutil.copyfile(raw_path, dst_in)

        # Results: same naming scheme used in save_output_image
        name, extension = original_filename.rsplit(".", 1)
        result_name = f"{name}_result.{extension}"
        dst_out = OK_RESULTS_DIR / result_name
        shutil.copyfile(result_path, dst_out)

        # Update last_ok
        shutil.copyfile(dst_out, LAST_OK_PATH)

        # Update last_ok por zona si viene identificador
        if zone_id:
            safe_zone = "".join(c for c in zone_id if c.isalnum() or c in "-_")
            if safe_zone:
                zone_last_ok = OK_DIR / f"last_ok_{safe_zone}.jpg"
                shutil.copyfile(dst_out, zone_last_ok)

    # Deteccion de manometro con YOLO (opcionalmente recorta)
    def detect_manometer(self, file_path: str, crop: bool = True):
        try:
            yolo_detector = YoloDetector(DETECT_MODEL_PATH)
            detected_image, bboxes = yolo_detector.process_image(file_path)
        except Exception as e:
            return {
                "code": 400,
                "message": f"Error in YOLO detection with model {DETECT_MODEL_PATH}: {e}",
                "data": "",
                "bboxes": np.zeros((0, 4), dtype=int),
            }

        if getattr(bboxes, "shape", (0,))[0] > 0:
            if crop:
                cropped_image = self.crop_image(detected_image, bboxes)

                # Resize and save cropped image
                cropped_image = cv2.resize(cropped_image, (640, 640))
                cv2.imwrite(file_path, cropped_image)

                return {
                    "code": 200,
                    "message": "Manometer detected and cropped",
                    "data": file_path,
                    "bboxes": bboxes.cpu().numpy().astype(int),
                }

            cv2.imwrite(file_path, detected_image)
            return {
                "code": 206,
                "message": "Manometer detected but not cropped",
                "data": file_path,
                "bboxes": bboxes.cpu().numpy().astype(int),
            }

        return {
            "code": 206,
            "message": "No manometers found",
            "data": file_path,
            "bboxes": bboxes.cpu().numpy().astype(int),
        }

    # Recorte del primer bbox con un margen pequeno
    def crop_image(self, detected_image, bboxes):
        h_img, w_img = detected_image.shape[:2]
        x1, y1, x2, y2 = bboxes[0].cpu().numpy().astype(int)

        bw, bh = x2 - x1, y2 - y1
        margin = 0.05
        dx = int(bw * margin)
        dy = int(bh * margin)

        x1m = max(0, x1 + dx)
        y1m = max(0, y1 + dy)
        x2m = min(w_img, x2 - dx)
        y2m = min(h_img, y2 - dy)

        return detected_image[y1m:y2m, x1m:x2m]

    # Lectura de puntos clave con modelo de "lecture"
    def read_manometer(self, detected_image_path: str):
        try:
            detection_lecture = YoloDetectorLecture(LECTURE_MODEL_PATH)
            points = detection_lecture.process_image(detected_image_path)
        except Exception as e:
            return {
                "code": 400,
                "message": f"Error in YOLO lecture with model {LECTURE_MODEL_PATH}: {e}",
                "data": "",
                "bboxes": np.zeros((0, 4), dtype=int),
            }

        if len(points) > 3:
            angle, max_angle, min_angle, points = self.compute_geometry(points)
            return {
                "code": 200,
                "message": "Regress angle calculated successfully",
                "data": {
                    "final_angle": angle,
                    "max_angle": max_angle,
                    "min_angle": min_angle,
                    "points": points,
                },
            }

        return {
            "code": 206,
            "message": "Cannot calculate regress angle, lecture is not completed",
            "data": detected_image_path,
            "bboxes": np.zeros((0, 4), dtype=int),
        }

    # Geometria de angulos usando base, punta, minimo y maximo
    def compute_geometry(self, points: dict):
        # Tip angle
        x0, y0 = points["base"]
        xt, yt = points["tip"]
        angle = math.degrees(math.atan2(yt - y0, xt - x0))
        angle = (angle + 360) % 360

        # Min angle
        xmin, ymin = points["minimum"]
        min_angle = math.degrees(math.atan2(ymin - y0, xmin - x0))
        min_angle = (min_angle + 360) % 360

        # Max angle
        xmax, ymax = points["maximum"]
        max_angle = math.degrees(math.atan2(ymax - y0, xmax - x0))
        max_angle = max_angle + 360

        return angle, max_angle, min_angle, points

    # Renderiza anotaciones y guarda la imagen final en temp
    def save_output_image(self, filename: str, segmented_img_path: str, angle: float, max_angle: float, min_angle: float, points: dict, bboxes):
        level = self.get_level_from_angle(angle, max_angle, min_angle)

        segmented_img = cv2.imread(segmented_img_path)
        segmented_img_rgb = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB)
        plt.imshow(segmented_img_rgb)

        x0, y0 = points["base"]
        xt, yt = points["tip"]
        xmin, ymin = points["minimum"]
        xmax, ymax = points["maximum"]

        # Keep the original color coding (needle/red, min/blue, max/green)
        plt.plot([x0, xt], [y0, yt], c="r", lw=3, alpha=0.6)
        plt.plot((x0, xmin), (y0, ymin), c="b", lw=3, alpha=0.6)
        plt.plot((x0, xmax), (y0, ymax), c="g", lw=3, alpha=0.6)

        plt.xlim(0, 640)
        plt.ylim(640, 0)
        plt.axis("off")

        bbox_props = dict(boxstyle="square,pad=0.3", ec="white", lw=2, fc="white", alpha=0.7)
        plt.text(123, 120, f"{level:.2f}{self.MEASURE_UNIT}", color="black", fontsize=20, fontweight="bold", ha="right", bbox=bbox_props,)

        name, extension = filename.rsplit(".", 1)
        result_name = f"{name}_result.{extension}"
        out_path = TEMP_DIR / result_name

        plt.savefig(str(out_path), bbox_inches="tight", pad_inches=0)
        plt.close()

        return {"code": 200, "message": "Image saved", "data": str(out_path), "bboxes": bboxes}

    # Convierte el angulo a nivel en unidades de la escala configurada
    def get_level_from_angle(self, angle: float, max_angle: float, min_angle: float) -> float:
        level = 100 * (angle - min_angle) / (max_angle - min_angle)
        level = np.clip(level, 0, 100)
        units = (level / 100) * (self.MEASURE_MAX - self.MEASURE_MIN) + self.MEASURE_MIN
        return round(float(units), 2)


if __name__ == "__main__":
    # Ejecucion local del servidor
    host = "0.0.0.0"
    port = 5000

    manometer_detect = ManometerDetect(host, port)
    manometer_detect.start()
