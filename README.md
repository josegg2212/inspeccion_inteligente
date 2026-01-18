# inspeccion_inteligente

Guia rapida para ejecutar el proyecto de inspeccion inteligente.

## Requisitos

- Python 3.8+ y pip
- Node.js + npm (necesario para el dashboard, por ejemplo en Raspberry Pi)

## Instalacion y ejecucion

1) Crear un entorno virtual con acceso a paquetes del sistema:

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
```

2) Clonar el repositorio y entrar:

```bash
git clone https://github.com/josegg2212/inspeccion_inteligente.git
cd inspeccion_inteligente
```

3) Instalar dependencias de Python:

```bash
pip install -r requirements.txt
```

4) (Opcional/Raspberry) Instalar Node.js y npm si no estan:

```bash
sudo apt install -y nodejs npm
```

5) Instalar dependencias del dashboard si hace falta:

```bash
cd src/dashboard
npm install
cd ../..
```

6) Revisar variables de entorno del dashboard:

Archivo: `src/dashboard/.env` (por defecto usa `http://0.0.0.0:5000` y `mqtt://0.0.0.0:1883`).

7) Levantar servicios y robot:

```bash
./run_services.sh
```

En otra terminal:

```bash
./run_robot.sh
```

## Dashboard

Acceso: `http://localhost:3000` (o `http://<IP_RASPI>:3000` desde otra maquina).

## Estructura del proyecto

- `example_images/`: imagenes de ejemplo para pruebas.
- `models/`: modelos y pesos (por ejemplo para YOLO y lectura de manometros).
- `src/api/`: API REST y logica de deteccion/lectura de manometros.
- `src/dashboard/`: servidor Node.js + frontend del dashboard.
- `src/logica_inspeccion/`: logica de robot y simulador.
- `src/mqtt/`: broker/cliente MQTT y configuracion simple.
- `run_services.sh`: levanta API, MQTT y dashboard.
- `run_robot.sh`: ejecuta el simulador/logica del robot.
