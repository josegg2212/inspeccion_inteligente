from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import libcamera
from picamera2 import Picamera2, Preview

from .cliente_api import ClienteAPI


@dataclass(frozen=True)
class ZonaInspeccion:
    id: str
    unit: str
    min_value: float
    max_value: float
    espera_s: float

    def metadata_dict(self) -> dict:
        return {
            "MEASURE_SCALE": {
                "UNIT": self.unit,
                "MIN": self.min_value,
                "MAX": self.max_value,
            }
        }


class Camara:
    """Preview QTGL + ScalerCrop (v2) + captura a JPG con OpenCV."""

    def __init__(self, vflip: bool = True) -> None:
        self.vflip = vflip
        self.cam: Optional[Picamera2] = None

    def start(self) -> None:
        self.cam = Picamera2()

        cfg = self.cam.create_preview_configuration(main={"size": (640, 480)})
        cfg["transform"] = libcamera.Transform(vflip=self.vflip)
        self.cam.configure(cfg)

        self.cam.start_preview(Preview.QTGL)
        self.cam.start()
        time.sleep(0.2)

        # Camera v2: evita el "mega zoom"
        self.cam.set_controls({"ScalerCrop": (0, 0, 3280, 2464)})

    def captura(self, output_path: Path) -> None:
        if self.cam is None:
            raise RuntimeError("Cámara no iniciada")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        frame = np.ascontiguousarray(self.cam.capture_array("main"))  # RGB
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(output_path), frame_bgr):
            raise RuntimeError(f"No se pudo guardar {output_path}")

    def stop(self) -> None:
        if self.cam is None:
            return
        try:
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


class RobotInspeccion:
    """Recorre zonas: captura + POST a la API + guarda resultado."""

    def __init__(
        self,
        zonas: List[ZonaInspeccion],
        camara: Camara,
        api: ClienteAPI,
        evidencias_dir: Path,
    ) -> None:
        self.zonas = zonas
        self.camara = camara
        self.api = api
        self.evidencias_dir = evidencias_dir

    def run(self) -> None:
        self.camara.start()
        try:
            while True:
                for z in self.zonas:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_in = self.evidencias_dir / f"{z.id}_{ts}.jpg"
                    img_out = self.evidencias_dir / f"{z.id}_{ts}_result.jpg"

                    self.camara.captura(img_in)

                    status, headers, out_bytes = self.api.procesar(img_in, z.metadata_dict())
                    if out_bytes:
                        img_out.write_bytes(out_bytes)
                        (self.evidencias_dir / "last_result.jpg").write_bytes(out_bytes)

                    (self.evidencias_dir / "last.jpg").write_bytes(img_in.read_bytes())

                    print(f"[{z.id}] status={status} msg='{headers.get('X-Message','')}' "
                          f"bboxes={headers.get('X-Bounding-Boxes','[]')}")

                    time.sleep(float(z.espera_s))
        except KeyboardInterrupt:
            print("Parado.")
        finally:
            self.camara.stop()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    evidencias = root / "data" / "evidencias"
    evidencias.mkdir(parents=True, exist_ok=True)

    zonas = [
        ZonaInspeccion("zona_A", "bar", 0.0, 10.0, 5.0),
        ZonaInspeccion("zona_B", "bar", 0.0, 16.0, 7.0),
    ]

    api = ClienteAPI(url_md="http://127.0.0.1:5000/MD")
    robot = RobotInspeccion(zonas=zonas, camara=Camara(vflip=True), api=api, evidencias_dir=evidencias)
    robot.run()


if __name__ == "__main__":
    main()
