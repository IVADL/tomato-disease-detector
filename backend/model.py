"""
The code was modified accordingly by taking some code 
from https://github.com/harimkang/food-image-classifier.
"""


from tensorflow.keras import backend
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import matplotlib.pyplot as plt
import numpy as np
import time
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # tensorflow logging off


class ClassificationModel:
    """
    [Tensorflow Imagenet Model]
    # Class created using model provided in keras.applications
    """

    def __init__(self, class_list, img_width, img_height, batch_size=16) -> None:

        backend.clear_session()

        self.class_list = class_list
        self.img_width, self.img_height = img_width, img_height
        # batch_size can be up to 16 based on GPU 4GB (not available for 32)
        self.batch_size = batch_size

        self.model = None

        self.train_data = None
        self.validation_data = None
        self.num_train_data = None

        self.day_now = time.strftime("%Y%m%d%H", time.localtime(time.time()))
        self.checkpointer = None
        self.csv_logger = None
        self.history = None
        self.model_name = "inception_v3"

    def generate_train_val_data(self, data_dir="dataset/train/"):
        """
        # Create an ImageDataGenerator by dividing the train and validation set
        # by 0.8/0.2 based on the train dataset folder.
        # train : 60600 imgs / validation : 15150 imgs
        """
        num_data = 0
        for root, dirs, files in os.walk(data_dir):
            if files:
                num_data += len(files)

        self.num_train_data = num_data
        _datagen = image.ImageDataGenerator(
            rescale=1.0 / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2,
        )
        self.train_data = _datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="training",
        )
        self.validation_data = _datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="validation",
        )

    def set_model(self, model_name="inception_v3"):
        """
        # This is a function that composes a model, and proceeds to compile.
        # [Reference] - https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_v3
        """
        if model_name == "inception_v3":
            self.model = InceptionV3(weights="imagenet", include_top=False)
        elif model_name == "mobilenet_v2":
            self.model = MobileNetV2(weights="imagenet", include_top=False)
            self.model_name = model_name
        x = self.model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.2)(x)
        pred = Dense(
            len(self.class_list),
            kernel_regularizer=regularizers.l2(0.005),
            activation="softmax",
        )(x)

        self.model = Model(inputs=self.model.input, outputs=pred)
        self.model.compile(
            optimizer=SGD(lr=0.0001, momentum=0.9),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return 1

    def train(self, epochs=10):
        """
        # Training-related environment settings (log, checkpoint) and training
        """
        train_samples = self.num_train_data * 0.8
        val_samples = self.num_train_data * 0.2

        os.makedirs(os.path.join("models", "checkpoint"), exist_ok=True)
        self.checkpointer = ModelCheckpoint(
            filepath=f"models/checkpoint/{self.model_name}_checkpoint_{self.day_not}.hdf5",
            verbose=1,
            save_best_only=True,
        )
        os.makedirs(os.path.join("logs", "training"), exist_ok=True)
        self.csv_logger = CSVLogger(
            f"logs/training/{self.model_name}_history_model_{self.day_now}.log"
        )

        self.history = self.model.fit_generator(
            self.train_data,
            steps_per_epoch=train_samples // self.batch_size,
            validation_data=self.validation_data,
            validation_steps=val_samples // self.batch_size,
            epochs=epochs,
            verbose=1,
            callbacks=[self.csv_logger, self.checkpointer],
        )

        self.model.save(f"models/{self.model_name}_model_{self.day_now}.hdf5")

        return self.history

    def evaluation(self, batch_size=16, data_dir="test/", steps=5):
        """
        # Evaluate the model using the data in data_dir as a test set.
        """
        if self.model is not None:
            test_datagen = image.ImageDataGenerator(rescale=1.0 / 255)
            test_generator = test_datagen.flow_from_directory(
                data_dir,
                target_size=(self.img_height, self.img_width),
                batch_size=batch_size,
                class_mode="categorical",
            )
            scores = self.model.evaluate_generator(test_generator, steps=steps)
            print("Evaluation data: {}".format(data_dir))
            print("%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))
        else:
            print("Model not found... : load_model or train plz")

    def prediction(self, image_data=None, img_path=None, show=False, save=False):
        """
        # Given a path for an image, the image is predicted and displayed through plt.
        """
        if img_path is not None:
            target_name = img_path.split(".")[0]
            target_name = target_name.split("/")[-1]
            save_time = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))

            img = image.load_img(
                img_path, target_size=(self.img_height, self.img_width)
            )
            img = image.img_to_array(img)
        elif image_data is not None:
            img = image_data
            # img = np.asarray(img)
        else:
            raise Exception("image binary data or image_path needed...")

        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255.0

        if self.model is not None:
            pred = self.model.predict(img)
            index = np.argmax(pred)
            self.class_list.sort()
            pred_value = self.class_list[index]
            if show:
                plt.imshow(img[0])
                plt.axis("off")
                plt.title("prediction: {}".format(pred_value))
                print("[Model Prediction] {}: {}".format(target_name, pred_value))
                plt.show()
                if save:
                    os.makedirs("results", exist_ok=True)
                    plt.savefig(
                        f"results/{self.model_name}_example_{target_name}_{save_time}.png"
                    )
            return pred_value
        else:
            print("Model not found... : load_model or train plz")
            return 0

    def load(self, model_path=None):
        """
        # If an already trained model exists, load it.
        """
        try:
            if model_path is None:
                model_path = "models/checkpoint/"
                os.makedirs(model_path, exist_ok=True)
                model_list = os.listdir(model_path)
                if model_list:
                    h5_list = [file for file in model_list if file.endswith(".hdf5")]
                    h5_list.sort()
                    backend.clear_session()
                    self.model = load_model(model_path + h5_list[-1], compile=False)
            else:
                backend.clear_session()
                self.model = load_model(model_path, compile=False)
                print("Model loaded...: ", model_path)

            self.model.compile(
                optimizer=SGD(lr=0.0001, momentum=0.9),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )
            return 1
        except:
            print("Model not found... : train plz")
            return 0

    def show_accuracy(self):
        """
        # Shows the accuracy graph of the training history.
        # TO DO: In the case of a loaded model, a function to find and display the graph is added
        """
        if self.history is not None:
            title = f"model_accuracy_{self.day_now}"
            plt.title(title)
            plt.plot(self.history.history["accuracy"])
            plt.plot(self.history.history["val_accuracy"])
            plt.ylabel("accuracy")
            plt.xlabel("epoch")
            plt.legend(["train_acc", "val_acc"], loc="best")
            plt.show()
            os.makedirs("results", exist_ok=True)
            plt.savefig(f"results/accuracy_{self.model_name}_model_{self.day_now}.png")
        else:
            print("Model not found... : load_model or train plz")

    def show_loss(self):
        """
        # Shows the loss graph of the training history.
        # TO DO: In the case of a loaded model, a function to find and display the graph is added
        """
        if self.history is not None:
            title = f"model_loss_{self.day_now}"
            plt.title(title)
            plt.plot(self.history.history["loss"])
            plt.plot(self.history.history["val_loss"])
            plt.ylabel("loss")
            plt.xlabel("epoch")
            plt.legend(["train_loss", "val_loss"], loc="best")
            plt.show()
            os.makedirs("results", exist_ok=True)
            plt.savefig(
                f"results/loss_{self.model_name}_model_{self.day_now}.png".format()
            )
        else:
            print("Model not found... : load_model or train plz")
