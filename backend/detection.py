from yolov5 import DetectorModel


def get_model(weights):
    model_detector = DetectorModel(weights)
    return model_detector