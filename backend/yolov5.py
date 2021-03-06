import torch
from numpy import random
from pathlib import Path
import cv2
import uuid

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import (
    check_img_size,
    non_max_suppression,
    scale_coords,
    set_logging,
)
from utils.plots import plot_one_box
from utils.torch_utils import select_device

# yolo_model = "custom"

# logging.info(f"YOLO model - {yolo_model}")

# weights = "./weights/yolov5s_tomato_3classes.pt"

# model = torch.hub.load("ultralytics/yolov5", yolo_model, path_or_model=weights)
# model = model.autoshape()


class DetectorModel:
    def __init__(self, weights, device="cpu"):

        self.weights = weights
        self.conf_thres = 0.5
        self.iou_thres = 0.45
        self.classes = None
        self.agnostic_nms = False

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

    def detect(self, source, image_size, save_path="./data/results/"):

        stride = int(self.model.stride.max())  # model stride
        image_size = check_img_size(image_size, s=stride)  # check img_size

        # Set Dataloader
        # vid_path, vid_writer = None, None
        dataset = LoadImages(source, img_size=image_size, stride=stride)

        # Run inference
        if self.device.type != "cpu":
            self.model(
                torch.zeros(1, 3, image_size, image_size)
                .to(self.device)
                .type_as(next(self.model.parameters()))
            )  # run once

        for path, img, im0s, vid_cap in dataset:
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
                p, s, im0, frame = path, "", im0s, getattr(dataset, "frame", 0)
                p = Path(p)  # to Path
                s += "%gx%g " % img.shape[2:]  # print string
                # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape
                    ).round()
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

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
                # Save results (image with detections)
                # image_name = f"{str(uuid.uuid4())}.png"
                # result_path = save_path + image_name
                # cv2.imwrite(result_path, im0)
                return im0


if __name__ == "__main__":
    detector_model = DetectorModel("backend/weights/yolov5s_tomato_3classes.pt")
    detector_model.detect("backend/data/", image_size=416)