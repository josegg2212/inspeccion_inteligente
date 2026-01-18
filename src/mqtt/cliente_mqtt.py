import json
import time
from datetime import datetime, timezone

import paho.mqtt.client as mqtt
from sense_hat import SenseHat

BROKER_HOST = "0.0.0.0"  
BROKER_PORT = 1883

TOPIC = "sensors/telemetry"
PUBLISH_EVERY_S = 2

QOS = 1
RETAIN = True

sense = SenseHat()

# Lee sensores de Sense HAT y genera payload normalizado
def read_telemetry():
    humidity = float(sense.get_humidity())
    temp_h = float(sense.get_temperature_from_humidity())
    temp_p = float(sense.get_temperature_from_pressure())
    temperature = temp_h 
    pressure = float(sense.get_pressure())
    if pressure == 0.0:
        time.sleep(0.05)
        pressure = float(sense.get_pressure())

    ts = datetime.now(timezone.utc).isoformat()
    return {
        "ts": ts,
        "temperature": round(temperature, 2),
        "humidity": round(humidity, 2),
        "pressure": round(pressure, 2),
    }

# Publicador MQTT: envia telemetria en intervalos fijos
def main():
    client = mqtt.Client(client_id="rpi-sensehat-pub", clean_session=True)
    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    client.loop_start()

    try:
        while True:
            payload = read_telemetry()
            client.publish(TOPIC, json.dumps(payload), qos=QOS, retain=RETAIN)
            print("Published:", payload)
            time.sleep(PUBLISH_EVERY_S)
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    main()
