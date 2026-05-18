import os, json, io
import numpy as np
import torch, torch.nn as nn
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn

with open("model/stage1_meta.json") as f:
    meta = json.load(f)

CLASSES         = meta["classes"]
NUM_CLASSES     = meta["num_classes"]
EMBED_DIM       = meta["embed_dim"]
IMG_SIZE        = meta["img_size"]
MEAN            = meta["mean"]
STD             = meta["std"]
TEMPERATURE     = meta["temperature"]
FUSED_THRESHOLD = meta["fused_threshold"]
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model    = models.resnet50(weights=None)
model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(EMBED_DIM, NUM_CLASSES))
ckpt     = torch.load("model/resnet50_best.pth", map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state"])
model    = model.to(DEVICE).eval()

_emb = {}
model.avgpool.register_forward_hook(
    lambda m,i,o: _emb.__setitem__("e", o.squeeze(-1).squeeze(-1).detach()))

transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

app = FastAPI(title="OOD Vision System")
app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health():
    return {"status": "ok", "model": "resnet50", "classes": CLASSES}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg","image/png","image/jpg"]:
        raise HTTPException(400, "Only JPEG/PNG supported")
    try:
        img    = Image.open(io.BytesIO(await file.read())).convert("RGB")
        arr    = np.array(img)
        tensor = transform(image=arr)["image"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(tensor)
        cal      = logits.cpu().numpy() / TEMPERATURE
        prob     = np.exp(cal) / np.exp(cal).sum()
        pred     = int(prob.argmax())
        conf_val = float(prob.max())
        fused    = 1 - conf_val
        is_ood   = bool(fused > FUSED_THRESHOLD)
        return JSONResponse({
            "predicted_class" : CLASSES[pred],
            "confidence"      : round(conf_val * 100, 2),
            "knn_score"       : 0.0,
            "entropy"         : 0.0,
            "fused_score"     : round(fused, 4),
            "fused_threshold" : round(FUSED_THRESHOLD, 4),
            "is_ood"          : is_ood,
            "status"          : "OOD" if is_ood else "In-Distribution"
        })
    except Exception as e:
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
