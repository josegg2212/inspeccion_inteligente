from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import libcamera
from picamera2 import Picamera2

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
    """Captura con PiCamera2 (como en la práctica)."""

    def __init__(
        self,
        main_size: tuple[int, int] = (1920, 1080),
        lores_size: tuple[int, int] = (640, 480),
        vflip: bool = True,
        hflip: bool = False,
    ) -> None:
        self.main_size = main_size
        self.lores_size = lores_size
        self.vflip = vflip
        self.hflip = hflip
        self.cam: Optional[Picamera2] = None

    def start(self) -> None:
        self.cam = Picamera2()

        cfg = self.cam.create_still_configuration(
            main={"size": self.main_size},
            lores={"size": self.lores_size},
            display="lores",
        )
        cfg["transform"] = libcamera.Transform(vflip=self.vflip, hflip=self.hflip)

        self.cam.configure(cfg)
        self.cam.start()
        time.sleep(0.2)  # warm-up corto

    def captura(self, output_path: Path) -> Path:
        if self.cam is None:
            raise RuntimeError("Cámara no iniciada. Llama a start().")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.cam.capture_file(str(output_path))
        return output_path

    def stop(self) -> None:
        if self.cam is not None:
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

    camara = Camara(
        main_size=(1920, 1080),
        lores_size=(640, 480),
        vflip=True,
        hflip=False,
    )
    api = ClienteAPI(url_md=url_md)
    robot = RobotInspeccion(zonas=zonas, camara=camara, api=api, evidencias_dir=evidencias_dir, loop=True)
    robot.run()


if __name__ == "__main__":
    main()
