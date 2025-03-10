from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights # standard model to customize
import uvicorn
import ImagePreprocessor

app = FastAPI(title="Image Classification API")


"""
Default app to improve replace RESNET with SOTA OCR models
"""

# Load pre-trained model on startup
@app.on_event("startup")
def load_model():
    global model
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()
    print("Model loaded successfully")

# Preprocess function
def preprocess_image(image: Image.Image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    return preprocess(image).unsqueeze(0)

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=415,
                detail="Unsupported media type"
            )
        
        # Read and process image
        image = Image.open(io.BytesIO(await file.read()))
        input_tensor = preprocess_image(image)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
        
        # Process results
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_class = torch.topk(probabilities, 5)
        
        return JSONResponse({
            "predictions": [
                {
                    "class": ResNet50_Weights.DEFAULT.meta["categories"][c],
                    "probability": round(p.item(), 4)
                }
                for p, c in zip(top_prob, top_class)
            ]
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    finally:
        await file.close()

if __name__ == "__main__":
    def main():
        uvicorn.run(app, host="0.0.0.0", port=8000)
    
    main()