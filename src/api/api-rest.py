"""Single-file FastAPI service for manometer detection + lecture.

- Exposes POST /MD that receives:
  - file: image
  - metadata: JSON file with MEASURE_SCALE {UNIT, MAX, MIN}
- Returns an annotated JPEG image.
- Adds headers:
  - X-Message
  - X-Bounding-Boxes (JSON list)

This file inlines the previous ApiRestWrapper so you only need one script.
"""

from __future__ import annotations

import json
import math
import os
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


class ApiRestWrapper:
    """Tiny wrapper around FastAPI + uvicorn.

    routes: [(url, methods, callback)]
    """

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


TEMPORAL_PATH = "./temp"

# Model weights
PROJECT_ROOT = Path(__file__).resolve().parents[2]  

DETECT_MODEL_PATH = str(PROJECT_ROOT / "models" / "manometer_detection_model.pt")
LECTURE_MODEL_PATH = str(PROJECT_ROOT / "models" / "manometer_lecture_model.pt")


class ManometerDetect:
    """FastAPI endpoint handler for manometer detection and lecture."""

    def __init__(self, host: str, port: int, endpoint: str) -> None:
        self.host = host
        self.port = port
        self.endpoint = endpoint

        self.MEASURE_UNIT = ""
        self.MEASURE_MAX = 0.0
        self.MEASURE_MIN = 0.0

        self.apirest_wrapper = ApiRestWrapper(self.host, self.port)
        self.apirest_wrapper.setup_routes([(endpoint, ["POST"], self.endpoint_callback)])

    def start(self):
        self.apirest_wrapper.run_app()

    async def endpoint_callback(
        self,
        file: UploadFile = File(...),
        metadata: UploadFile = File(...),
    ):
        """Receive image + metadata, process and return annotated image."""

        metadata_content = await metadata.read()
        try:
            config_data = json.loads(metadata_content)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid metadata JSON: {e}")

        try:
            response = self.process_image(file, config_data)

            if response["code"] == 400:
                raise HTTPException(status_code=400, detail=response["message"])

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

    def process_image(self, file: UploadFile, config_data: dict):
        # Store metadata parameters
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

        if os.path.exists(TEMPORAL_PATH):
            shutil.rmtree(TEMPORAL_PATH)
        os.makedirs(TEMPORAL_PATH, exist_ok=True)

        # Save input file to disk
        file_path = os.path.join(TEMPORAL_PATH, file.filename)
        with open(file_path, "wb") as f:
            f.write(file.file.read())

        # 1) Detect & crop manometer
        response = self.detect_manometer(file_path)
        if response["code"] != 200:
            return response
        detected_image_path = response["data"]
        bboxes = response["bboxes"]

        # 2) Regress angle
        response = self.regress_angle(detected_image_path)
        if response["code"] != 200:
            return response

        angle = response["data"]["final_angle"]
        max_angle = response["data"]["max_angle"]
        min_angle = response["data"]["min_angle"]
        points = response["data"]["points"]

        # 3) Save final annotated image
        return self.save_output_image(
            file.filename,
            detected_image_path,
            angle,
            max_angle,
            min_angle,
            points,
            bboxes,
        )

    def detect_manometer(self, file_path: str, crop: bool = True):
        try:
            yolo_detector = YoloDetector(DETECT_MODEL_PATH)
            detected_image, bboxes = yolo_detector.process_image(file_path)
        except Exception as e:
            return {
                "code": 400,
                "message": f"Error in YOLO detection with model {DETECT_MODEL_PATH}: {e}",
                "data": "",
            }

        if getattr(bboxes, "shape", (0,))[0] > 0:
            if crop:
                cropped_image = self.crop_image(detected_image, bboxes)

                # Resize and save cropped image (lecture model expects 640x640)
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
            "message": "Not manometers found",
            "data": file_path,
            "bboxes": bboxes.cpu().numpy().astype(int),
        }

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

    def regress_angle(self, detected_image_path: str):
        try:
            detection_lecture = YoloDetectorLecture(LECTURE_MODEL_PATH)
            points = detection_lecture.process_image(detected_image_path)
        except Exception as e:
            return {
                "code": 400,
                "message": f"Error in YOLO lecture with model {LECTURE_MODEL_PATH}: {e}",
                "data": "",
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
        }

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

    def save_output_image(
        self,
        filename: str,
        segmented_img_path: str,
        angle: float,
        max_angle: float,
        min_angle: float,
        points: dict,
        bboxes,
    ):
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
        plt.text(
            123,
            120,
            f"{level:.2f}{self.MEASURE_UNIT}",
            color="black",
            fontsize=20,
            fontweight="bold",
            ha="right",
            bbox=bbox_props,
        )

        name, extension = filename.rsplit(".", 1)
        result_name = f"{name}_result.{extension}"
        out_path = os.path.join(TEMPORAL_PATH, result_name)

        plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close()

        return {"code": 200, "message": "Image saved", "data": out_path, "bboxes": bboxes}

    def get_level_from_angle(self, angle: float, max_angle: float, min_angle: float) -> float:
        level = 100 * (angle - min_angle) / (max_angle - min_angle)
        level = np.clip(level, 0, 100)
        units = (level / 100) * (self.MEASURE_MAX - self.MEASURE_MIN) + self.MEASURE_MIN
        return round(float(units), 2)


if __name__ == "__main__":
    host = "0.0.0.0"
    port = 5000
    endpoint = "/MD"

    manometer_detect = ManometerDetect(host, port, endpoint)
    manometer_detect.start()
