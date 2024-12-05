from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
from io import BytesIO
from app.utils import load_model, preprocess_image

app = FastAPI()

model = load_model()

fruit_classes = ['apple', 'durian', 'grape', 'strawberry', 'dragon fruit']

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

        image_bytes = await file.read()
        img = Image.open(BytesIO(image_bytes))

        processed_image = preprocess_image(img)

        prediction = model.predict(processed_image)
        predicted_class_idx = np.argmax(prediction, axis=1)[0]
        predicted_class = fruit_classes[predicted_class_idx]

        ripeness_score = prediction[0][predicted_class_idx]  
        ripeness_status = "Ripe" if ripeness_score > 0.5 else "Unripe"

        return JSONResponse(content={
            "Nama Buah": predicted_class,
            "Rippenes": ripeness_status,
            "Persentase Rippenes": float(ripeness_score) * 100  
        })

    except HTTPException as http_err:
        raise http_err  
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Internal Server Error: {str(e)}"})
