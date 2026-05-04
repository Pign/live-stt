const TARGET_SR = 16000;

const $toggle = document.getElementById("toggle");
const $status = document.getElementById("status");
const $lang = document.getElementById("lang");
const $level = document.getElementById("level");
const $transcript = document.getElementById("transcript");

let ws = null;
let audioCtx = null;
let stream = null;
let workletNode = null;
let sourceNode = null;
let resampleRatio = 1;
let resampleResidual = 0;
let recording = false;

const finals = []; // [{seq, text}]
let partial = "";

function render() {
  const finalText = finals.map((f) => f.text).join(" ");
  $transcript.textContent = "";
  if (finalText) {
    $transcript.appendChild(document.createTextNode(finalText));
  }
  if (partial) {
    if (finalText) $transcript.appendChild(document.createTextNode(" "));
    const span = document.createElement("span");
    span.className = "partial";
    span.textContent = partial;
    $transcript.appendChild(span);
  }
}

// Linear resample from inputRate -> 16 kHz with a fractional residual carried
// across buffers so frame boundaries don't drift.
function resampleTo16k(int16, inputRate) {
  if (inputRate === TARGET_SR) return int16;
  const ratio = inputRate / TARGET_SR;
  const outLen = Math.floor((int16.length - resampleResidual) / ratio);
  const out = new Int16Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const srcIdx = resampleResidual + i * ratio;
    const i0 = Math.floor(srcIdx);
    const frac = srcIdx - i0;
    const s0 = int16[i0] || 0;
    const s1 = int16[i0 + 1] || s0;
    out[i] = (s0 + (s1 - s0) * frac) | 0;
  }
  const consumed = resampleResidual + outLen * ratio;
  resampleResidual = consumed - Math.floor(consumed);
  // Drop everything we consumed; residual is sub-sample.
  return out;
}

async function start() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
  } catch (e) {
    $status.textContent = "Micro refusé : " + e.message;
    return;
  }

  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  await audioCtx.audioWorklet.addModule("/static/pcm-worklet.js");

  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";

  await new Promise((resolve, reject) => {
    ws.onopen = resolve;
    ws.onerror = reject;
  });

  ws.send(JSON.stringify({ type: "config", language: $lang.value }));

  ws.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.type === "partial") {
      partial = msg.text;
      render();
    } else if (msg.type === "final") {
      partial = "";
      if (msg.text) finals.push({ seq: msg.seq, text: msg.text });
      render();
    }
  };
  ws.onclose = () => { $status.textContent = "Connexion fermée."; };

  resampleRatio = audioCtx.sampleRate / TARGET_SR;
  resampleResidual = 0;

  sourceNode = audioCtx.createMediaStreamSource(stream);
  workletNode = new AudioWorkletNode(audioCtx, "pcm-worklet");
  workletNode.port.onmessage = (ev) => {
    const { pcm, rms } = ev.data;
    $level.style.width = Math.min(100, rms * 300) + "%";
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const int16 = new Int16Array(pcm);
    const resampled = resampleTo16k(int16, audioCtx.sampleRate);
    if (resampled.length > 0) ws.send(resampled.buffer);
  };
  sourceNode.connect(workletNode);
  // Worklet doesn't need to feed the destination, but Chrome requires the
  // graph to be connected for processing to run reliably.
  workletNode.connect(audioCtx.destination);
  // Mute the loopback so the user doesn't hear themselves.
  const gain = audioCtx.createGain();
  gain.gain.value = 0;
  workletNode.disconnect(audioCtx.destination);
  workletNode.connect(gain).connect(audioCtx.destination);

  recording = true;
  $toggle.classList.add("recording");
  $toggle.textContent = "Arrêter";
  $status.textContent = `Écoute… (entrée ${audioCtx.sampleRate} Hz → 16 kHz)`;
}

async function stop() {
  recording = false;
  $toggle.classList.remove("recording");
  $toggle.textContent = "Démarrer";
  $status.textContent = "Arrêté.";
  $level.style.width = "0%";

  if (ws && ws.readyState === WebSocket.OPEN) {
    try { ws.send(JSON.stringify({ type: "flush" })); } catch {}
    setTimeout(() => ws && ws.close(), 500);
  }
  if (workletNode) workletNode.disconnect();
  if (sourceNode) sourceNode.disconnect();
  if (audioCtx) await audioCtx.close();
  if (stream) stream.getTracks().forEach((t) => t.stop());
  ws = null; workletNode = null; sourceNode = null; audioCtx = null; stream = null;
}

$toggle.addEventListener("click", () => {
  if (recording) stop(); else start();
});

$lang.addEventListener("change", () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "config", language: $lang.value }));
  }
});
