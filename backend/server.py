from fastapi import FastAPI, File
from fastapi.logger import logger
from starlette.responses import Response

import uuid

from detection import get_model as get_det_model

import io
from PIL import Image

detector_model = get_det_model("./weights/yolov5s_tomato_7classes.pt")

app = FastAPI(
    title="Plant Disease Detector",
    description="Plant Disease Detector using DL Models",
    version="0.1.0",
)


# @app.post("/classification")
# def get_predict_disease_classification(file: bytes = File(...)):
#     """"""
#     image = Image.open(io.BytesIO(file)).convert("RGB")
#     pred = cls_model.prediction(image_data=image)

#     return Response(content=pred, status_code=200)


@app.post("/detection")
def get_predict_disease_detector(file: bytes = File(...)):
    name = f"./data/{str(uuid.uuid4())}.png"

    logger.info("get image")
    image = Image.open(io.BytesIO(file))#.convert("RGB")

    # opencv save -> bgr / PIL Image - open RGB ==> need convert bgr to rgb
    b, g, r = image.split()
    image = Image.merge("RGB", (r, g, b))

    image.save(name)

    logger.info("save image")
    try:
        converted_img = detector_model.detect("./data/", image_size=416)
        converted_img = Image.fromarray(converted_img)
        bytes_io = io.BytesIO()
        converted_img.save(bytes_io, format="PNG")
    except Exception as e:
        logger.warning(e)

    return Response(bytes_io.getvalue(), media_type="image/png")
    # return StreamingResponse(content=converted_img, media_type="image/png")
