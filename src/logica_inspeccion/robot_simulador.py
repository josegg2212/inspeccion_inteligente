from __future__ import annotations

import time
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import libcamera
from picamera2 import Picamera2, Preview

from cliente_api import ClienteAPI
try:
    from sense_hat import SenseHat
except Exception:
    SenseHat = None

# Clase que representa una zona de inspección con sus parámetros
@dataclass(frozen=True)
class ZonaInspeccion:
    id: str
    unit: str
    min_value: float
    max_value: float
    espera_s: float

    def metadata_dict(self) -> dict:
        # Metadata esperada por la API para convertir angulo a unidades
        return {
            "ZONE_ID": self.id,
            "MEASURE_SCALE": {
                "UNIT": self.unit,
                "MIN": self.min_value,
                "MAX": self.max_value,
            }
        }


# Clase que maneja la cámara PiCamera2
class Camara:
    def __init__(self, preview_size: Tuple[int, int] = (640, 480), sensor_full_crop: Tuple[int, int, int, int] = (0, 0, 3280, 2464), vflip: bool = True, hflip: bool = False, preview: bool = True, preview_mode: str = "qtgl") -> None:
        self.preview_size = preview_size
        self.sensor_full_crop = sensor_full_crop
        self.vflip = vflip
        self.hflip = hflip
        self.preview = preview
        self.preview_mode = preview_mode.lower()
        self.cam: Optional[Picamera2] = None

    def start(self) -> None:
        # Inicializa camara y configura preview/crop
        self.cam = Picamera2()

        cfg = self.cam.create_preview_configuration(main={"size": self.preview_size})
        cfg["transform"] = libcamera.Transform(vflip=self.vflip, hflip=self.hflip)
        self.cam.configure(cfg)

        self.cam.start_preview(Preview.QTGL)

        self.cam.start()
        time.sleep(0.2)

        try:
            self.cam.set_controls({"ScalerCrop": self.sensor_full_crop})
        except Exception as e:
            print(f"[Camara] No se pudo aplicar ScalerCrop={self.sensor_full_crop}: {e}")

    def captura(self, output_path: Path) -> Path:
        # Captura un frame y lo guarda como JPG
        if self.cam is None:
            raise RuntimeError("Cámara no iniciada. Llama a start().")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        frame = self.cam.capture_array("main")  
        frame = np.ascontiguousarray(frame)

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(output_path), frame_bgr):
            raise RuntimeError(f"No se pudo guardar imagen en {output_path}")

        return output_path

    def stop(self) -> None:
        # Apagado seguro de la camara
        if self.cam is not None:
            try:
                if self.preview:
                    self.cam.stop_preview()
            except Exception:
                pass
            try:
                self.cam.stop()
            except Exception:
                pass
            try:
                self.cam.close()
            except Exception:
                pass
        self.cam = None


# Clase principal que simula el robot de inspección
class RobotInspeccion:
    def __init__(self, zonas: List[ZonaInspeccion], camara: Camara, api: ClienteAPI, evidencias_dir: Path, loop: bool = True, guardar_ultima: bool = True) -> None:
        self.zonas = zonas
        self.camara = camara
        self.api = api
        self.evidencias_dir = evidencias_dir
        self.loop = loop
        self.guardar_ultima = guardar_ultima
        self.running = False

    def run(self) -> None:
        # Loop principal: captura -> envia API -> guarda evidencias
        self.running = True
        self.camara.start()
        sense = SenseHat() if SenseHat else None

        try:
            while self.running:
                for zona in self.zonas:
                    if not self.running:
                        break

                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_in = self.evidencias_dir / f"{zona.id}_{ts}.jpg"
                    img_out = self.evidencias_dir / f"{zona.id}_{ts}_result.jpg"

                    # Cuenta atras en Sense HAT antes de capturar
                    if sense:
                        for n in (3, 2, 1):
                            sense.show_letter(str(n))
                            time.sleep(1)
                        sense.clear()

                    # Captura local
                    self.camara.captura(img_in)

                    if self.guardar_ultima:
                        (self.evidencias_dir / "last.jpg").write_bytes(img_in.read_bytes())

                    # POST a la API de deteccion/lectura
                    status, headers, out_bytes = self.api.procesar(img_in, zona.metadata_dict())

                    # Guarda resultado si la respuesta trae imagen
                    if out_bytes:
                        img_out.parent.mkdir(parents=True, exist_ok=True)
                        img_out.write_bytes(out_bytes)

                        if self.guardar_ultima:
                            (self.evidencias_dir / "last_result.jpg").write_bytes(out_bytes)

                        # Si la lectura fue correcta (HTTP 200), guarda tambien en /ok
                        if status == 200:
                            ok_dir = self.evidencias_dir / "ok"
                            ok_dir.mkdir(parents=True, exist_ok=True)
                            (ok_dir / img_out.name).write_bytes(out_bytes)
                            (ok_dir / "last_ok_result.jpg").write_bytes(out_bytes)

                    # Info por consola
                    msg = headers.get("X-Message", "")
                    bboxes = headers.get("X-Bounding-Boxes", "[]")
                    print(f"[{zona.id}] status={status} msg='{msg}' bboxes={bboxes}")

                    time.sleep(max(0.0, float(zona.espera_s)))

                if not self.loop:
                    break

        except KeyboardInterrupt:
            print("Parado por teclado.")
        finally:
            self.camara.stop()
            self.running = False

    def stop(self) -> None:
        self.running = False


def main() -> None:
    # Configuracion base del simulador
    project_root = Path(__file__).resolve().parents[2]
    evidencias_dir = project_root / "data" / "evidencias"
    evidencias_dir.mkdir(parents=True, exist_ok=True)

    url_endpoint_api = "http://0.0.0.0:5000/process_camera_image"

    zonas = [
        ZonaInspeccion("zona_A", "bar", 0.0, 1.0, espera_s=10.0),
        ZonaInspeccion("zona_B", "bar", 0.0, 25.0, espera_s=10.0),
    ]

    camara = Camara(preview_size=(640, 480), sensor_full_crop=(0, 0, 3280, 2464), vflip=True, hflip=True, preview=True, preview_mode="qtgl")
    api = ClienteAPI(url_endpoint=url_endpoint_api)
    robot = RobotInspeccion(zonas=zonas, camara=camara, api=api, evidencias_dir=evidencias_dir, loop=True, guardar_ultima=True)
    robot.run()


if __name__ == "__main__":
    main()
