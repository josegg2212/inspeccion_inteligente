/* server.js */
require("dotenv").config();

const express = require("express");
const http = require("http");
const { Server } = require("socket.io");
const mqtt = require("mqtt");

const PORT = Number(process.env.PORT || 3000);
const IMAGE_URL = process.env.IMAGE_URL;
const MQTT_URL = process.env.MQTT_URL;
const MQTT_TOPIC = process.env.MQTT_TOPIC || "sensors/telemetry";

if (!IMAGE_URL) {
  console.error("Missing IMAGE_URL in .env");
  process.exit(1);
}
if (!MQTT_URL) {
  console.error("Missing MQTT_URL in .env");
  process.exit(1);
}

const app = express();
app.use(express.static("public"));

app.get("/api/health", (req, res) => {
  res.json({ ok: true, mqttTopic: MQTT_TOPIC });
});

// Proxy endpoint to fetch the image server-side (avoids CORS)
app.get("/api/image", async (req, res) => {
  try {
    const resp = await fetch(IMAGE_URL, { cache: "no-store" });
    if (!resp.ok) {
      return res.status(502).json({ error: "Failed to fetch image", status: resp.status });
    }

    const contentType = resp.headers.get("content-type") || "application/octet-stream";
    res.setHeader("Content-Type", contentType);
    res.setHeader("Cache-Control", "no-store");

    // Stream the image
    const arrayBuffer = await resp.arrayBuffer();
    res.send(Buffer.from(arrayBuffer));
  } catch (err) {
    console.error("Image fetch error:", err);
    res.status(500).json({ error: "Image fetch error" });
  }
});

const server = http.createServer(app);
const io = new Server(server, {
  cors: { origin: "*" }
});

// ---- MQTT ----
const mqttOptions = {
  clientId: process.env.MQTT_CLIENT_ID || `dashboard-${Math.random().toString(16).slice(2)}`,
  username: process.env.MQTT_USERNAME || undefined,
  password: process.env.MQTT_PASSWORD || undefined,
  reconnectPeriod: 2000
};

let latestTelemetry = null;
let mqttConnected = false;

function toNumber(value) {
  if (typeof value === "number") return value;
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed) return null;
    const n = Number(trimmed);
    if (!Number.isNaN(n)) return n;
  }
  return null;
}

function parseTelemetry(payload) {
  // Accept JSON: {"temperature": 23.5, "humidity": 45, "pressure": 1012} or plain number: "23.5"
  const text = payload.toString("utf8").trim();
  if (!text) return null;

  // Try JSON
  try {
    const obj = JSON.parse(text);
    if (obj && typeof obj === "object") {
      const temperature = toNumber(obj.temperature ?? obj.temp ?? obj.T ?? obj.value);
      const humidity = toNumber(obj.humidity ?? obj.hum ?? obj.H);
      const pressure = toNumber(obj.pressure ?? obj.press ?? obj.P);
      if (temperature !== null || humidity !== null || pressure !== null) {
        return { temperature, humidity, pressure };
      }
    }
  } catch (_) {
    // Not JSON, ignore
  }

  // Try plain number
  const n = Number(text);
  if (!Number.isNaN(n)) return { temperature: n, humidity: null, pressure: null };

  return null;
}

const mqttClient = mqtt.connect(MQTT_URL, mqttOptions);

mqttClient.on("connect", () => {
  mqttConnected = true;
  console.log("MQTT connected:", MQTT_URL);
  mqttClient.subscribe(MQTT_TOPIC, { qos: 0 }, (err) => {
    if (err) console.error("MQTT subscribe error:", err);
    else console.log("Subscribed to:", MQTT_TOPIC);
  });
  io.emit("mqtt_status", { connected: true });
});

mqttClient.on("reconnect", () => {
  mqttConnected = false;
  io.emit("mqtt_status", { connected: false });
});

mqttClient.on("close", () => {
  mqttConnected = false;
  io.emit("mqtt_status", { connected: false });
});

mqttClient.on("error", (err) => {
  mqttConnected = false;
  console.error("MQTT error:", err.message);
  io.emit("mqtt_status", { connected: false, error: err.message });
});

mqttClient.on("message", (topic, payload) => {
  const telemetry = parseTelemetry(payload);
  if (!telemetry) return;

  latestTelemetry = telemetry;
  io.emit("telemetry", {
    temperature: telemetry.temperature,
    humidity: telemetry.humidity,
    pressure: telemetry.pressure,
    topic,
    ts: Date.now()
  });
});

// ---- Websocket clients ----
io.on("connection", (socket) => {
  socket.emit("mqtt_status", { connected: mqttConnected });
  if (latestTelemetry !== null) {
    socket.emit("telemetry", {
      temperature: latestTelemetry.temperature,
      humidity: latestTelemetry.humidity,
      pressure: latestTelemetry.pressure,
      topic: MQTT_TOPIC,
      ts: Date.now()
    });
  }
});

server.listen(PORT, () => {
  console.log(`Dashboard running on http://localhost:${PORT}`);
});
