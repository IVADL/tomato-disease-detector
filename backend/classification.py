from model import ClassificationModel


def get_label(file_path):
    f = open(file_path, "r")
    lines = f.readlines()
    label = []
    for line in lines:
        label.append(line.strip("\n"))
    f.close()

    return label


def get_model(model_path):
    label = get_label("./util/label.txt")
    model = ClassificationModel(
        class_list=label,
        img_width=256,
        img_height=256,
    )
    status = model.load(model_path)
    if not status:
        raise Exception("model load failed...")

    return model