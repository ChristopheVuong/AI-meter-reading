from pathlib import Path
from PIL import Image

import torch
import pandas as pd

from ultralytics.utils.plotting import save_one_box
from ultralytics import YOLO
from ultralytics.utils import LOGGER, TryExcept, ops

from roboflow import Roboflow


class ResultMeters:
    """
    A class for storing and manipulating inference results containing dial meters.

    This class encapsulates among other the functionality for handling dial meters
    and classification results from YOLO models. it is a composition with the original
    results class in the package ultralytics.

    Attributes:
        result (ultralytics.engine.Results): A result containing the bounding boxes and associated probabilities for classification tasks.
        detection_precision (int): A flag indicating the degree of certainty of the position of bounding boxes.
        filename (str): Path to the image file.

    Methods:
        save_meters_crop(save_dir, file_name=None): Saves cropped detection (containing meters) images to specified directory (one per dial) following some criterion.
        set_detection_precision(flag=0): Set the precision of the detection from the scale from -1 to 1 (no, maybe, sure).
        set_filename(file_name): Set the filename of the image.
        show: Shows annotated results on screen.
    """

    def __init__(self, result):
        """
        Result for
        """
        self.result = result
        self.detection_precision = None
        self.filename = None

    def save_meters_crop(self, save_dir, file_name=None):
        """
        Saves cropped detection (containing meters) images to specified directory (one per dial) following some criterion.

        This method saves cropped images of detected objects to a specified directory. Each crop is saved in a
        subdirectory named after the object's class, with the filename based on the input file_name.

        Args:
            save_dir (str | Path): Directory path where cropped images will be saved (flag -1, 0 or 1)
            file_name (str | Path): Base filename for the saved cropped images. Default is Path("im.jpg").

        Notes:
            - This method does not support Classify or Oriented Bounding Box (OBB) tasks.
            - Crops are saved as 'save_dir/class_name/file_name.jpg'.
            - The method will create necessary subdirectories if they don't exist.
            - Original image is copied before cropping to avoid modifying the original.
        """
        cropped = False
        if self.result.probs is not None:
            LOGGER.warning("WARNING ⚠️ Classify task do not support `save_crop`.")
            return
        if self.result.obb is not None:
            LOGGER.warning("WARNING ⚠️ OBB task do not support `save_crop`.")
            return
        if file_name is None:
            file_name = Path(self.result.path).stem
        # It is ensured that the boxes are not empty!
        max_conf = 0
        box1_max_conf = None  # only class meters
        # box0_min_conf = None
        for d in self.result.boxes:
            confidence = d.conf
            if int(d.cls) == 1 and confidence > max_conf:
                max_conf = confidence
                box1_max_conf = d

        if box1_max_conf is not None:
            cropped = True
            self.set_detection_precision(1)
            crop = save_one_box(
                box1_max_conf.xyxy,
                self.result.orig_img.copy(),
                file=Path(save_dir) / f"{Path(file_name)}-res.jpg",
                BGR=True,
            )
        else:
            self.set_detection_precision(-1)
            file = Path(save_dir) / f"{Path(file_name)}-res.jpg"
            file.parent.mkdir(parents=True, exist_ok=True)  # make directory
            Image.fromarray(self.result.orig_img.copy()).save(
                file, quality=95, subsampling=0
            )  # save RGB
            LOGGER.warning("WARNING ⚠️ No meters detected.")
        self.set_filename(file_name)
        return cropped

    def set_detection_precision(self, flag=0):
        """
        Set the precision of the detection from the scale from -1 to 1 (no, maybe, sure)
        """
        self.detection_precision = flag

    def set_filename(self, file_name):
        """
        Set the filename of the image
        """
        self.filename = file_name

    def show(self, result):
        """
        Use the show method of Result class to show the resulting annotated image
        """
        self.result.show()

# Suppose the pictures and test folder are in Challenge_Suez
PATH = "/content/drive/MyDrive"
CHALLENGE_NAME = "/AI-meter-reading"
PATH_PROJECT = PATH + CHALLENGE_NAME
FOLDER_IMAGE_PATH = PATH_PROJECT + "/pictures"

if __name__ == '__main__':
    # Do not hardcode
    rf = Roboflow(api_key="YPr8z8exI5EnxDegn87q")  # available until late October
    project = rf.workspace("ai-meter-reading").project("ai-meter-reading")
    version = project.version(1)
    dataset = version.download("yolov11")  # in the main folder

    model = YOLO("yolo11n.pt")

    model.train(data=f"{dataset.location}/data.yaml", epochs=100)

    results = model.predict(
        source=FOLDER_IMAGE_PATH, project="pictures_annotated", stream=True
    )  # generator in order to avoid memory leak
