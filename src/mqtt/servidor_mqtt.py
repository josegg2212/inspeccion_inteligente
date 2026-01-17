import shutil
import subprocess
from pathlib import Path

HOST = "0.0.0.0"
PORT = 1883

def main():
    mosquitto = shutil.which("mosquitto")
    if mosquitto is None:
        print("ERROR: mosquitto no está instalado. Instala con: sudo apt install mosquitto")
        return

    # Creamos un config mínimo en la misma carpeta (si no existe)
    conf_path = Path(__file__).with_name("mosquitto_simple.conf")
    if not conf_path.exists():
        conf_path.write_text(
            f"""listener {PORT} {HOST}
allow_anonymous true
persistence false
log_dest stdout
""",
            encoding="utf-8",
        )

    # Arranca Mosquitto en primer plano (Ctrl+C para parar)
    subprocess.run([mosquitto, "-c", str(conf_path), "-v"], check=False)

if __name__ == "__main__":
    main()
