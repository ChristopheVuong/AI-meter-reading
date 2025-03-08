import pandas as pd
import numpy as np
import os
import json
import logging
import yaml
from pathlib import Path
from typing import List, Dict, Union, Optional

# YOLO format handler, tensorflow

class TensorFlowLabels:
    """
    A class for handling TensorFlow image labels and bounding boxes.
    """

    def __init__(self, logging_level=logging.INFO):
        """
        Initialize the TensorFlowLabels class.

        Args:
            logging_level: The logging level to use
        """
        # Configure logging
        logging.basicConfig(
            level=logging_level, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def load_bounding_boxes(self, bbs_file: str) -> pd.DataFrame:
        """
        Load bounding box data from a TensorFlow labels file.

        Args:
            bbs_file: Path to the file containing bounding box coordinates

        Returns:
            DataFrame containing the parsed bounding box data
        """
        if not os.path.exists(bbs_file):
            raise FileNotFoundError(f"Bounding boxes file not found: {bbs_file}")

        try:
            bbs_df = pd.read_csv(
                bbs_file,
                header=None,
                names=["filename", "xmin", "ymin", "xmax", "ymax"],
            )

            if bbs_df.empty:
                self.logger.warning(f"Bounding boxes file is empty: {bbs_file}")

            return bbs_df
        except Exception as e:
            self.logger.error(f"Failed to load bounding boxes file: {str(e)}")
            raise

    def load_labels(self, labels_file: str) -> pd.DataFrame:
        """
        Load label data from a TensorFlow labels file.

        Args:
            labels_file: Path to the file containing label index mappings

        Returns:
            DataFrame containing the parsed label data
        """
        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"Labels file not found: {labels_file}")

        try:
            labels_df = pd.read_csv(labels_file, header=None, names=["ID", "index"])

            if labels_df.empty:
                self.logger.warning(f"Labels file is empty: {labels_file}")

            return labels_df
        except Exception as e:
            self.logger.error(f"Failed to load labels file: {str(e)}")
            raise

    def convert_to_paddleocr(
        self,
        bbs_file: str,
        labels_file: str,
        img_dir: str,
        output_file: str,
        overwrite: bool = False,
    ) -> List[str]:
        """
        Convert TensorFlow format annotations to PaddleOCR format and save to file.

        Args:
            bbs_file: Path to file containing bounding box coordinates
            labels_file: Path to file containing label index mappings
            img_dir: Directory containing the images
            output_file: Path to save the converted annotations
            overwrite: Whether to overwrite existing output file

        Returns:
            List of annotation strings in PaddleOCR format
        """
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        annotations = []
        try:
            # Load the data
            bbs_df = self.load_bounding_boxes(bbs_file)
            labels_df = self.load_labels(labels_file)

            if bbs_df.empty:
                return []

            for idx, row in bbs_df.iterrows():
                try:
                    img_filename = str(row.filename).split("_")[0]
                    if not img_filename:
                        self.logger.warning(f"Invalid filename: {row.filename}")
                        continue

                    img_path = os.path.join(img_dir, img_filename + ".jpg")
                    if not os.path.exists(img_path):
                        self.logger.warning(f"Image not found: {img_path}")
                        continue

                    # Perform VLOOKUP-like operation to get the label
                    label_row = labels_df[labels_df["ID"] == img_filename]
                    if not label_row.empty:
                        index = str(label_row.iloc[0]["index"])
                    else:
                        self.logger.warning(f"No label found for image: {img_filename}")
                        index = ""

                    annotation = self._convert_annotation(row, index)

                    annotations.append(f"{img_path}\t{json.dumps([annotation])}")

                except Exception as e:
                    self.logger.error(f"Error processing row {idx}: {str(e)}")
                    continue

            self._save_annotations(annotations, output_file, overwrite)

            self.logger.info(f"Successfully processed {len(annotations)} annotations")

        except Exception as err:
            self.logger.error(f"Failed to process files: {str(err)}")
            raise err from None

        return annotations

    def _convert_annotation(self, row: pd.Series, index: str) -> Dict:
        """
        Convert a single annotation from TensorFlow format to PaddleOCR format.

        Args:
            row: DataFrame row containing bounding box coordinates
            index: Label index for the annotation

        Returns:
            Dictionary containing the annotation in PaddleOCR format
        """
        # Convert bounding box coordinates to PaddleOCR format and ensure they're floats
        xmin, ymin, xmax, ymax = map(
            float, [row["xmin"], row["ymin"], row["xmax"], row["ymax"]]
        )
        # if non-validation of bounding box, permutate the coordinates in the clockwise direction
        points = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]

        annotation = {"transcription": index, "points": points}
        return annotation

    @staticmethod
    def _save_annotations(annotations: List[str], output_file: str, overwrite: bool) -> None:
        """
        Save the annotations to a file.
        
        Args:
            annotations: List of annotation strings in PaddleOCR format
            output_file: Path to save the annotations
            overwrite: Whether to overwrite an existing file
        """
        if os.path.exists(output_file) and not overwrite:
            raise FileExistsError(f"Output file already exists and overwrite=False: {output_file}")
        
        with open(output_file, 'w') as out_file:
            for annotation in annotations:
                out_file.write(annotation + "\n")

    @staticmethod
    def validate_bounding_box(xmin: float, ymin: float, xmax: float, ymax: float) -> bool:
        """
        Validate that bounding box coordinates are valid.
        
        Args:
            xmin: Minimum x coordinate
            ymin: Minimum y coordinate
            xmax: Maximum x coordinate
            ymax: Maximum y coordinate
            
        Returns:
            True if the bounding box is valid, False otherwise
        """
        # Check that coordinates are in the correct order
        if xmin >= xmax or ymin >= ymax:
            return False
            
        # Check that coordinates are positive
        if xmin < 0 or ymin < 0 or xmax < 0 or ymax < 0:
            return False
            
        return True


# Example usage
# access from a Roboflow project
### provide with parameters : tensorflow to paddleocr simple dataset, or yolo to tensorflow (what others?)
### use argparse to provide the parameters
### by default in argparse it looks in src/configs
if __name__ == "__main__":
    # Load YAML file
    project_dir = Path(__file__).parent.parent
    config_path = project_dir / "configs" / "dataset" / "tensorflow_to_simpleDataset.yaml"
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
