from fastapi import FastAPI, File, BackgroundTasks
from fastapi.logger import logger
from fastapi.responses import Response

import cv2

import asyncio

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

tasks = {}


@app.post("/detection/image")
def post_predict_disease_detector_image(file: bytes = File(...)):

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
async def post_predict_disease_detector_video(background_tasks: BackgroundTasks):
    logger.info(f"Post Success Video")
    name = f"/var/lib/assets/detect1.mp4"
    logger.info(f"file: {name}")

    video_path = "/var/lib/assets/video1.mp4"
    cap = cv2.VideoCapture(video_path)

    background_tasks.add_task(
        detector_model.detect, cap, image_size=416, video=True, save_path=name
    )
    # asyncio.create_task(
    #     detector_model.detect(cap, image_size=416, video=True, save_path=name)
    # )


@app.get("/detection/video/status")
async def get_predict_disease_detector_video():
    status, progress, save_path = detector_model.get_status()

    return {"status": status, "progress": progress, "save_path": save_path}
