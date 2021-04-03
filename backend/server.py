from fastapi import FastAPI, File
from fastapi.logger import logger
from fastapi.responses import Response, FileResponse

import uuid
import tempfile
import cv2
import os

from detection import get_model as get_det_model

import io
import numpy as np
from PIL import Image

detector_model = get_det_model("./weights/yolov5s_tomato_7classes.pt")

app = FastAPI(
    title="Tomato Disease Detector",
    description="Plant Disease Detector using DL Models",
    version="0.1.0",
)


@app.post("/detection/image")
async def post_predict_disease_detector_image(file: bytes = File(...)):

    logger.info("get image")
    image = Image.open(io.BytesIO(file))  # .convert("RGB")
    open_cv_image = np.array(image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    converted_img = detector_model.detect(open_cv_image, image_size=416)
    converted_img = Image.fromarray(converted_img)
    bytes_io = io.BytesIO()
    converted_img.save(bytes_io, format="PNG")

    return Response(bytes_io.getvalue(), media_type="image/png")


@app.post("/detection/video")
def post_predict_disease_detector_video(file: bytes = File(...)):
    file_id = str(uuid.uuid4())
    name = f"./data/results/{file_id}.mp4"

    logger.info(f"file: {file}")

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(file)

    cap = cv2.VideoCapture(tfile.name)
    detector_model.detect(cap, image_size=416, video=True, save_path=name)
    video_file = open(name, "rb")
    video_bytes = video_file.read()

    return Response(video_bytes, media_type="video/mp4")


# @app.post("/detection/video")
# def post_predict_disease_detector_video():

#     file_id = str(uuid.uuid4())
#     name = f"/var/lib/assets/{file_id}.mp4"

#     logger.info(file)
#     cap = cv2.VideoCapture(file)

#     try:
#         detector_model.detect(cap, image_size=416, video=True, save_path=name)
#     except Exception as e:
#         logger.warning(e)

#     return Response(status_code=202, content=name)


# @app.get("/detection/video/status")
# def get_predict_disease_detector_status():
#     try:
#         status = detector_model.status
#         progress = detector_model.progress
#         return Response(status_code=202, content=dict(status=status, progress=progress))
#     except Exception as e:
#         logger.warning(e)


# @app.get("/detection/video/result")
# def get_predict_disease_detector_video():
#     try:
#         status = detector_model.status
#         if status == "Success" and os.path.isfile(detector_model.save_dir):
#             # video_result = open(f"./data/results/{job_id}.mp4", mode="rb")
#             return FileResponse(detector_model.save_dir, media_type="video/mp4")
#         else:
#             return dict(status="Error")
#     except Exception as e:
#         logger.warning(e)
