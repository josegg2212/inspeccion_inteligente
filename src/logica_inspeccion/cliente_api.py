from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import requests

# Simple HTTP client para llamar a la API REST de inspección
class ClienteAPI:
    def __init__(self, url_endpoint: str, timeout_s: int = 60) -> None:
        self.url_endpoint = url_endpoint
        self.timeout_s = timeout_s
        # Session para reutilizar conexion HTTP
        self.session = requests.Session()

    def procesar(self, image_path: Path, metadata: dict) -> tuple[int, dict, Optional[bytes]]:
        # Empaqueta imagen + metadata en multipart y llama a la API
        metadata_bytes = json.dumps(metadata).encode("utf-8")

        with open(image_path, "rb") as f:
            files = {
                "file": (image_path.name, f, "image/jpeg"),
                "metadata": ("metadata.json", metadata_bytes, "application/json"),
            }
            r = self.session.post(self.url_md, files=files, timeout=self.timeout_s)

        # Solo devuelve bytes si la respuesta es una imagen
        img_bytes = None
        if r.content and r.headers.get("content-type", "").startswith("image/"):
            img_bytes = r.content

        return r.status_code, dict(r.headers), img_bytes
