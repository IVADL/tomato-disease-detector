from fastapi import FastAPI, File
from starlette.responses import Response

from classification import get_model

import io
from PIL import Image

model = get_model("./models/mobilenet_v2_checkpoint_202101281638.hdf5")

app = FastAPI(
    title="Plant Disease Detector",
    description="Plant Disease Detector using DL Models",
    version="0.1.0",
)


@app.post("/classification")
def get_predict_disease_result(file: bytes = File(...)):
    """"""
    image = Image.open(io.BytesIO(file)).convert("RGB")
    pred = model.prediction(image_data=image)

    return Response(content=pred, status_code=200)
