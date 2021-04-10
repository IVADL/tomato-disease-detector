import torch
from numpy import random
import cv2
import numpy as np
from fastapi.logger import logger

from models.experimental import attempt_load
from utils.general import (
    check_img_size,
    non_max_suppression,
    scale_coords,
    set_logging,
)
from utils.plots import plot_one_box
from utils.torch_utils import select_device


class DetectorModel:
    def __init__(self, weights, device="cpu"):

        self.weights = weights
        self.conf_thres = 0.5
        self.iou_thres = 0.45
        self.classes = None
        self.agnostic_nms = False
        self.status = "Pending"
        self.progress = 0.0
        self.save_path = None

        # Initialize
        set_logging()
        self.device = select_device(device)
        self.half = device != "cpu"  # half precision only supported on CUDA
        if self.half:
            self.model.half()  # to FP16

        # Load model
        self.model = attempt_load(weights, map_location=device)  # load FP32 model

        # Get names and colors
        self.names = (
            self.model.module.names
            if hasattr(self.model, "module")
            else self.model.names
        )
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

    def get_status(self):
        return self.status, self.progress, self.save_path

    def detect(self, source, image_size, save=False, video=False, save_path=None):

        logger.info("Start Detection")

        stride = int(self.model.stride.max())  # model stride
        image_size = check_img_size(image_size, s=stride)  # check img_size

        # Set Dataloader
        vid_path, vid_writer = None, None
        dataset = LoadData(source, img_size=image_size, stride=stride, video=video)

        self.status = "In Progress"
        self.save_path = save_path

        # Run inference
        if self.device.type != "cpu":
            self.model(
                torch.zeros(1, 3, image_size, image_size)
                .to(self.device)
                .type_as(next(self.model.parameters()))
            )  # run once

        for img, im0s, vid_cap, nframes in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = self.model(img, augment=False)[0]

            # Apply NMS
            pred = non_max_suppression(
                pred,
                self.conf_thres,
                self.iou_thres,
                classes=self.classes,
                agnostic=self.agnostic_nms,
            )

            # Process detections
            for det in pred:  # detections per image
                im0, frame = im0s, getattr(dataset, "frame", 0)
                print(f"frame: {frame}/{nframes}")
                self.progress = (frame / nframes) * 100
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape
                    ).round()
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        label = f"{self.names[int(cls)]} {conf:.2f}"
                        plot_one_box(
                            xyxy,
                            im0,
                            label=label,
                            color=self.colors[int(cls)],
                            line_thickness=5,
                        )

            if save or video:
                if not video:
                    cv2.imwrite(f"./data/results/test1.png", im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = "mp4v"  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h)
                        )
                    vid_writer.write(im0)
        logger.info("Finish Detection")
        self.progress = 100
        self.status = "Success"

        if not video:
            return im0


class LoadData:  # for inference
    def __init__(self, data, img_size=640, stride=32, video=False):

        self.img_size = img_size
        self.stride = stride

        self.cap = data
        if video:
            self.mode = "video"
            self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame = 0
        else:
            self.mode = "image"
            self.nframes = 1

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nframes:
            raise StopIteration

        if self.mode == "video":
            # Read video
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nframes:  # last video
                    raise StopIteration
                else:
                    ret_val, img0 = self.cap.read()

            self.frame += 1

        else:
            # Read image
            img0 = self.cap
            assert img0 is not None, "Image is None"
        self.count += 1

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img, img0, self.cap, self.nframes

    def __len__(self):
        return self.nframes  # number of files


def letterbox(
    img,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return img, ratio, (dw, dh)