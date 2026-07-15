import torch
from torchvision import transforms
from PIL import Image
import io
import sys
import os
import time

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.resnet_manual import MiniResNet

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "resnet_fruit.pth"

CLASS_NAMES = ["apple", "durian", "banana"]

CLASS_EMOJI = {
    "apple": "🍎",
    "durian": "🟢",
    "banana": "🍌"
}

NUM_CLASSES = len(CLASS_NAMES)

# ============================================================
# LOAD MODEL
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MiniResNet(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("====================================")
print(f"Model loaded: {MODEL_PATH}")
print(f"Device: {device}")
print(f"Classes: {CLASS_NAMES}")
print("====================================")

# ============================================================
# IMAGE TRANSFORM
# ============================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(title="Fruit Detection AI Web App")

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Fruit AI Detector</title>

<style>
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    min-height: 100vh;
    font-family: 'Segoe UI', Arial, sans-serif;
    background: linear-gradient(135deg, #0f172a, #1e3a8a, #0ea5e9);
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 24px;
}

.card {
    width: 100%;
    max-width: 540px;
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(18px);
    border-radius: 26px;
    padding: 34px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.4);
    border: 1px solid rgba(255,255,255,0.1);
    color: white;
}

.header {
    text-align: center;
    margin-bottom: 26px;
}

.header h1 {
    font-size: 30px;
    color: #38bdf8;
    margin-bottom: 8px;
}

.header p {
    color: #cbd5f5;
    font-size: 14px;
}

.badge-row {
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 24px;
}

.badge {
    background: rgba(56,189,248,0.15);
    border: 1px solid #38bdf8;
    color: #7dd3fc;
    padding: 8px 14px;
    border-radius: 999px;
    font-size: 14px;
    font-weight: 700;
}

.upload-area {
    border: 2px dashed #38bdf8;
    background: rgba(14,165,233,0.08);
    border-radius: 18px;
    padding: 34px 20px;
    text-align: center;
    cursor: pointer;
    position: relative;
    transition: 0.2s;
    margin-bottom: 18px;
}

.upload-area:hover {
    background: rgba(14,165,233,0.18);
    transform: translateY(-2px);
}

.upload-area input {
    position: absolute;
    inset: 0;
    opacity: 0;
    width: 100%;
    height: 100%;
    cursor: pointer;
}

.upload-icon {
    font-size: 46px;
    margin-bottom: 10px;
}

#previewBox {
    display: none;
    text-align: center;
    margin-bottom: 18px;
}

#preview {
    width: 210px;
    height: 210px;
    object-fit: cover;
    border-radius: 18px;
    border: 3px solid #38bdf8;
}

.button-row {
    display: grid;
    grid-template-columns: 1fr 110px;
    gap: 10px;
    margin-bottom: 18px;
}

button {
    border: none;
    border-radius: 14px;
    padding: 14px;
    font-size: 15px;
    font-weight: 800;
    cursor: pointer;
    transition: 0.2s;
}

#btn {
    background: linear-gradient(135deg, #38bdf8, #2563eb);
    color: white;
}

#resetBtn {
    background: rgba(255,255,255,0.1);
    color: white;
}

#result {
    display: none;
}

.loading {
    text-align: center;
    padding: 18px;
    color: #cbd5f5;
}

.spinner {
    width: 18px;
    height: 18px;
    border: 3px solid #1e3a8a;
    border-top-color: #38bdf8;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    display: inline-block;
    margin-right: 8px;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

.result-card {
    background: rgba(56,189,248,0.1);
    border: 1px solid #38bdf8;
    border-radius: 18px;
    padding: 24px;
    text-align: center;
}

.result-emoji {
    font-size: 64px;
    margin-bottom: 10px;
}

.result-label {
    font-size: 28px;
    font-weight: 900;
    color: #38bdf8;
    text-transform: capitalize;
}

.stats {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-top: 16px;
}

.stat {
    background: rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 14px;
}

.stat-value {
    font-size: 18px;
    font-weight: 900;
}

.stat-label {
    font-size: 12px;
    color: #cbd5f5;
}
</style>
</head>

<body>
<div class="card">

    <div class="header">
        <h1>🍎 Fruit AI Detector</h1>
        <p>Upload a fruit image and AI will predict it instantly</p>
    </div>

    <div class="badge-row">
        <span class="badge">Apple</span>
        <span class="badge">Durian</span>
        <span class="badge">Banana</span>
    </div>

    <div class="upload-area" id="uploadArea">
        <input type="file" id="fileInput" accept="image/*" onchange="handleFile(this)">
        <div class="upload-icon">📁</div>
        <p><strong>Click to upload image</strong></p>
    </div>

    <div id="previewBox">
        <img id="preview">
    </div>

    <div class="button-row">
        <button id="btn" onclick="predict()" disabled>Detect</button>
        <button id="resetBtn" onclick="resetForm()">Reset</button>
    </div>

    <div id="result"></div>
</div>

<script>
function handleFile(input){
    const file = input.files[0];
    if(!file) return;

    document.getElementById("preview").src = URL.createObjectURL(file);
    document.getElementById("previewBox").style.display = "block";
    document.getElementById("uploadArea").style.display = "none";
    document.getElementById("btn").disabled = false;
}

async function predict(){
    const file = document.getElementById("fileInput").files[0];
    const result = document.getElementById("result");

    result.style.display = "block";
    result.innerHTML = "<div class='loading'>Loading...</div>";

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("/predict", {
        method: "POST",
        body: formData
    });

    const data = await res.json();

    const emoji = {
        apple: "🍎",
        durian: "🟢",
        banana: "🍌"
    }[data.prediksi] || "❓";

    result.innerHTML = `
    <div class="result-card">
        <div class="result-emoji">${emoji}</div>
        <div class="result-label">${data.prediksi}</div>

        <div class="stats">
            <div class="stat">
                <div class="stat-value">${data.keyakinan}</div>
                <div class="stat-label">Confidence</div>
            </div>
            <div class="stat">
                <div class="stat-value">${data.latensi_ms} ms</div>
                <div class="stat-label">Latency</div>
            </div>
        </div>
    </div>`;
}

function resetForm(){
    location.reload();
}
</script>

</body>
</html>
"""


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    tensor = transform(image).unsqueeze(0).to(device)

    start = time.time()

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, dim=1)
        confidence = confidence.item()
        pred = pred.item()
        THRESHOLD = 0.6  # adjust if needed
        if confidence < THRESHOLD:
            label = "others"
        else:
            label = CLASS_NAMES[pred]
        
    latency = (time.time() - start) * 1000

    return {
        "prediksi": label,
        "keyakinan": f"{confidence * 100:.1f}%",
        "latensi_ms": f"{latency:.1f}"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)