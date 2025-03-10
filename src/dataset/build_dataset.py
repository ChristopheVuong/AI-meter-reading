# Write class for general dataset even dataloader that can then be exposed to either YOLO or PaddleOCR
# abstract class dataset

# Roboflow build dataset
from roboflow import Roboflow
import yaml
from utils import TensorFlowLabels
# read the keys from the YAML file
# the keys are the API keys for the Roboflow account
# find the file in 
with open("configs/dataset/roboflow.yml", "r") as file:
    keys = yaml.load(file, Loader=yaml.FullLoader)
# name of the project
# and version of the project, and also what kind of download we want for YOLO and also paddleocr
rf = Roboflow(api_key=keys["api_key"])
project = rf.workspace(keys["workspace"]).project(keys["project_name"])
version = project.version(keys["project_version"])
dataset = version.download(keys["download_type"])
# loading in dataset rather than storing it in the current working directory
# the dataset is downloaded in the current working directory

"""
Download dataset
If needed convert tensorflow dataset to PaddleOCR (static method as well?)
"""

# see utils for the format of the labels
config_path = "configs" / "dataset" / "tensorflow_to_simpleDataset.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
print(config)
# Check required keys in config
required_keys = ["bbs_file", "labels_file", "img_dir", "output_file"]
for key in required_keys:
    if key not in config:
        raise ValueError(f"Missing required config key: {key}")
# Initialize TensorFlowLabels class
tf_labels = TensorFlowLabels()
# Convert TensorFlow annotations to PaddleOCR format
annotations = tf_labels.convert_to_paddleocr(
    bbs_file=config["bbs_file"],
    labels_file=config["labels_file"],
    img_dir=config["img_dir"],
    output_file=config["output_file"],
    overwrite=config.get("overwrite", False),
)