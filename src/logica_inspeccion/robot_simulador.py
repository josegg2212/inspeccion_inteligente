from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2

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
    """Captura simple con OpenCV VideoCapture (como en muchas prácticas)."""

    def __init__(self, device_index: int = 0, width: int = 1280, height: int = 720) -> None:
        self.device_index = device_index
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None

    def start(self) -> None:
        self.cap = cv2.VideoCapture(self.device_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara (index={self.device_index})")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def captura(self, output_path: Path) -> Path:
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Cámara no iniciada. Llama a start().")

        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("No se pudo capturar frame.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        ok2 = cv2.imwrite(str(output_path), frame)
        if not ok2:
            raise RuntimeError(f"No se pudo guardar imagen en {output_path}")
        return output_path

    def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()
        self.cap = None


class RobotInspeccion:
    """Simula el robot: recorre zonas y en cada una captura + llama a la API."""

    def __init__(
        self,
        zonas: List[ZonaInspeccion],
        camara: Camara,
        api: ClienteAPI,
        evidencias_dir: Path,
        loop: bool = True,
    ) -> None:
        self.zonas = zonas
        self.camara = camara
        self.api = api
        self.evidencias_dir = evidencias_dir
        self.loop = loop
        self.running = False

    def run(self) -> None:
        self.running = True
        self.camara.start()

        try:
            while self.running:
                for zona in self.zonas:
                    if not self.running:
                        break

                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_in = self.evidencias_dir / f"{zona.id}_{ts}.jpg"
                    img_out = self.evidencias_dir / f"{zona.id}_{ts}_result.jpg"

                    # 1) Captura
                    self.camara.captura(img_in)

                    # 2) POST a la API
                    status, headers, out_bytes = self.api.procesar(img_in, zona.metadata_dict())

                    # 3) Guarda resultado si viene imagen
                    if out_bytes:
                        img_out.parent.mkdir(parents=True, exist_ok=True)
                        img_out.write_bytes(out_bytes)

                    # 4) Info por consola (simple)
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
    # raíz del proyecto: .../src/logica_inspeccion/robot_simulador.py -> sube 2 niveles
    project_root = Path(__file__).resolve().parents[2]
    evidencias_dir = project_root / "data" / "evidencias"

    # Si la API corre en la misma Raspberry:
    url_md = "http://127.0.0.1:5000/MD"
    # Si accedes a una API en otra máquina: url_md = "http://IP:5000/MD"

    zonas = [
        ZonaInspeccion("zona_A", "bar", 0.0, 10.0, espera_s=5.0),
        ZonaInspeccion("zona_B", "bar", 0.0, 16.0, espera_s=7.0),
    ]

    camara = Camara(device_index=0)
    api = ClienteAPI(url_md=url_md)
    robot = RobotInspeccion(zonas=zonas, camara=camara, api=api, evidencias_dir=evidencias_dir, loop=True)
    robot.run()


if __name__ == "__main__":
    main()
