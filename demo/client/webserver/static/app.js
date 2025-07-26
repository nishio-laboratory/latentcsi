const ws = new WebSocket("ws://localhost:8123/ws");

ws.onmessage = (e) => {
  console.log("raw data:", e.data, "typeof:", typeof e.data);
  document.getElementById("image").src = "data:image/png;base64," + e.data;
};

document.getElementById("slider").oninput = (e) => {
  fetch("/control/slider", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ value: parseFloat(e.target.value) })
  });
};


function sendCommand(endpoint) {
    fetch(endpoint, { method: "POST" })
}
