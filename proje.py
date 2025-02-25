from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# FastAPI uygulaması başlat
app = FastAPI()

# Eğitilmiş modeli yükle
model_path = "C:/Users/enisf/clip_model_final"
model = CLIPModel.from_pretrained(model_path)
processor = CLIPProcessor.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

@app.post("/predict/")
async def predict(file: UploadFile = File(...), text: str = "Bu bir askı sapandır"):
    image = Image.open(file.file).convert("RGB")

    # Görsel ve metni modele ver
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)

    # Skor hesapla
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    return {"text": text, "score": probs[0][0].item()}

# Uygulamayı başlatmak için
# uvicorn dosya_adı:app --reload