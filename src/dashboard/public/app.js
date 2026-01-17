/* public/app.js */

const socket = io();

const btnLoad = document.getElementById("btnLoad");
const imgView = document.getElementById("imgView");
const imgSkeleton = document.getElementById("imgSkeleton");

const mqttDot = document.getElementById("mqttDot");
const mqttText = document.getElementById("mqttText");

const tempNumber = document.getElementById("tempNumber");
const tempTs = document.getElementById("tempTs");
const humidityNumber = document.getElementById("humidityNumber");
const pressureNumber = document.getElementById("pressureNumber");
const topicPill = document.getElementById("topicPill");

const canvas = document.getElementById("tempChart");
const ctx = canvas.getContext("2d");

const history = []; // {ts, value}
const maxPoints = 60;

function setMqttStatus(connected, error) {
  mqttDot.style.background = connected ? "var(--good)" : "var(--bad)";
  mqttText.textContent = connected ? "MQTT: connected" : `MQTT: disconnected${error ? " (" + error + ")" : ""}`;
}

function formatTime(ts) {
  const d = new Date(ts);
  return d.toLocaleString();
}

async function loadImage() {
  imgSkeleton.style.display = "grid";
  imgView.style.display = "none";

  // cache-bust query param
  const url = `/api/image?ts=${Date.now()}`;
  imgView.onload = () => {
    imgSkeleton.style.display = "none";
    imgView.style.display = "block";
  };
  imgView.onerror = () => {
    imgSkeleton.textContent = "Failed to load image";
  };
  imgView.src = url;
}

btnLoad.addEventListener("click", loadImage);

// Auto-load once
loadImage();

// ----- Socket events -----
socket.on("mqtt_status", (data) => {
  setMqttStatus(Boolean(data.connected), data.error);
});

socket.on("telemetry", (data) => {
  const temperature = Number(data.temperature ?? data.value);
  const humidity = Number(data.humidity);
  const pressure = Number(data.pressure);

  const hasTemp = Number.isFinite(temperature);
  const hasHumidity = Number.isFinite(humidity);
  const hasPressure = Number.isFinite(pressure);
  if (!hasTemp && !hasHumidity && !hasPressure) return;

  const ts = data.ts || Date.now();
  topicPill.textContent = `topic: ${data.topic || "—"}`;
  tempTs.textContent = formatTime(ts);

  if (hasTemp) {
    tempNumber.textContent = temperature.toFixed(1);
    history.push({ ts, value: temperature });
    while (history.length > maxPoints) history.shift();
    drawChart();
  }

  if (hasHumidity) {
    humidityNumber.textContent = humidity.toFixed(1);
  }

  if (hasPressure) {
    pressureNumber.textContent = pressure.toFixed(1);
  }
});

// ----- Simple canvas chart -----
function drawChart() {
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  // Background grid
  ctx.globalAlpha = 1;
  ctx.lineWidth = 1;
  ctx.strokeStyle = "rgba(255,255,255,0.10)";
  for (let i = 1; i <= 4; i++) {
    const y = (h * i) / 5;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  if (history.length < 2) return;

  const values = history.map(p => p.value);
  let minV = Math.min(...values);
  let maxV = Math.max(...values);

  // Add padding
  const pad = Math.max(0.5, (maxV - minV) * 0.15);
  minV -= pad;
  maxV += pad;
  if (minV === maxV) { minV -= 1; maxV += 1; }

  // Plot line
  ctx.lineWidth = 3;
  ctx.strokeStyle = "rgba(124,92,255,0.95)";
  ctx.beginPath();

  history.forEach((p, i) => {
    const x = (i / (history.length - 1)) * (w - 20) + 10;
    const y = h - ((p.value - minV) / (maxV - minV)) * (h - 20) - 10;

    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });

  ctx.stroke();

  // Latest point
  const last = history[history.length - 1];
  const x = ( (history.length - 1) / (history.length - 1) ) * (w - 20) + 10;
  const y = h - ((last.value - minV) / (maxV - minV)) * (h - 20) - 10;

  ctx.fillStyle = "rgba(46,204,113,0.95)";
  ctx.beginPath();
  ctx.arc(x, y, 5, 0, Math.PI * 2);
  ctx.fill();

  // Min/Max labels
  ctx.fillStyle = "rgba(255,255,255,0.65)";
  ctx.font = "14px ui-sans-serif, system-ui";
  ctx.fillText(`max ${maxV.toFixed(1)}°C`, 12, 18);
  ctx.fillText(`min ${minV.toFixed(1)}°C`, 12, h - 10);
}
