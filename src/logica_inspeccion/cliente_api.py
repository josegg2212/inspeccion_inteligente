from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import requests


class ClienteAPI:
    """Cliente para tu endpoint POST /MD (multipart):
    - file: imagen
    - metadata: JSON (como archivo)
    Devuelve:
    - status_code
    - headers dict
    - bytes de imagen resultado (si viene imagen)
    """

    def __init__(self, url_md: str, timeout_s: int = 60) -> None:
        self.url_md = url_md
        self.timeout_s = timeout_s
        self.session = requests.Session()

    def procesar(self, image_path: Path, metadata: dict) -> tuple[int, dict, Optional[bytes]]:
        metadata_bytes = json.dumps(metadata).encode("utf-8")

        with open(image_path, "rb") as f:
            files = {
                "file": (image_path.name, f, "image/jpeg"),
                "metadata": ("metadata.json", metadata_bytes, "application/json"),
            }
            r = self.session.post(self.url_md, files=files, timeout=self.timeout_s)

        img_bytes = None
        if r.content and r.headers.get("content-type", "").startswith("image/"):
            img_bytes = r.content

        return r.status_code, dict(r.headers), img_bytes
