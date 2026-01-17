from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

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
    """Preview estilo libcamera-hello (4:3) + captura still (4:3) sin romper el preview."""

    def __init__(
        self,
        # PREVIEW 4:3 para que NO recorte (igual que libcamera-hello)
        preview_size: tuple[int, int] = (640, 480),
        # STILL 4:3 para mantener el mismo campo de visión que el preview
        still_size: tuple[int, int] = (2592, 1944),  # 4:3 típico (v2). Puedes bajar si quieres.
        vflip: bool = True,
        hflip: bool = False,
        preview: bool = True,
        preview_mode: str = "qt",  # "qt" | "drm" | "null"
    ) -> None:
        self.preview_size = preview_size
        self.still_size = still_size
        self.vflip = vflip
        self.hflip = hflip
        self.preview = preview
        self.preview_mode = preview_mode.lower()
        self.cam: Optional[Picamera2] = None

    def start(self) -> None:
        self.cam = Picamera2()

        # Preview config (4:3) -> se ve como libcamera-hello
        preview_cfg = self.cam.create_preview_configuration(
            main={"size": self.preview_size},
        )
        preview_cfg["transform"] = libcamera.Transform(vflip=self.vflip, hflip=self.hflip)
        self.cam.configure(preview_cfg)

        if self.preview:
            try:
                if self.preview_mode == "qt":
                    self.cam.start_preview(Preview.QT)
                elif self.preview_mode == "drm":
                    self.cam.start_preview(Preview.DRM)
                else:
                    self.cam.start_preview(Preview.NULL)
            except Exception as e:
                print(f"[Camara] No se pudo iniciar preview ({self.preview_mode}): {e}")

        self.cam.start()
        time.sleep(0.2)

        # Autofocus continuo si existe (Camera Module 3). Si no, no pasa nada.
        try:
            self.cam.set_controls({"AfMode": 2})
        except Exception:
            pass

    def captura(self, output_path: Path) -> Path:
        if self.cam is None:
            raise RuntimeError("Cámara no iniciada. Llama a start().")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Still config (también 4:3) para que la foto tenga el MISMO FOV que el preview
        still_cfg = self.cam.create_still_configuration(
            main={"size": self.still_size},
        )
        still_cfg["transform"] = libcamera.Transform(vflip=self.vflip, hflip=self.hflip)

        self.cam.switch_mode_and_capture_file(still_cfg, str(output_path))
        return output_path

    def stop(self) -> None:
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


class RobotInspeccion:
    """Simula el robot: recorre zonas y en cada una captura + llama a la API."""

    def __init__(
        self,
        zonas: List[ZonaInspeccion],
        camara: Camara,
        api: ClienteAPI,
        evidencias_dir: Path,
        loop: bool = True,
        guardar_ultima: bool = True,
    ) -> None:
        self.zonas = zonas
        self.camara = camara
        self.api = api
        self.evidencias_dir = evidencias_dir
        self.loop = loop
        self.guardar_ultima = guardar_ultima
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

                    if self.guardar_ultima:
                        (self.evidencias_dir / "last.jpg").write_bytes(img_in.read_bytes())

                    # 2) POST a la API
                    status, headers, out_bytes = self.api.procesar(img_in, zona.metadata_dict())

                    # 3) Guarda resultado si viene imagen
                    if out_bytes:
                        img_out.parent.mkdir(parents=True, exist_ok=True)
                        img_out.write_bytes(out_bytes)

                        if self.guardar_ultima:
                            (self.evidencias_dir / "last_result.jpg").write_bytes(out_bytes)

                    # 4) Info por consola
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
    project_root = Path(__file__).resolve().parents[2]
    evidencias_dir = project_root / "data" / "evidencias"
    evidencias_dir.mkdir(parents=True, exist_ok=True)

    url_md = "http://127.0.0.1:5000/MD"

    zonas = [
        ZonaInspeccion("zona_A", "bar", 0.0, 10.0, espera_s=5.0),
        ZonaInspeccion("zona_B", "bar", 0.0, 16.0, espera_s=7.0),
    ]

    camara = Camara(
        preview_size=(640, 480),        # como libcamera-hello
        still_size=(2592, 1944),        # 4:3 (mismo FOV que el preview)
        vflip=True,
        hflip=False,
        preview=True,
        preview_mode="qt",
    )

    api = ClienteAPI(url_md=url_md)
    robot = RobotInspeccion(
        zonas=zonas,
        camara=camara,
        api=api,
        evidencias_dir=evidencias_dir,
        loop=True,
        guardar_ultima=True,
    )
    robot.run()


if __name__ == "__main__":
    main()
