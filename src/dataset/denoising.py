import cv2
import numpy as np
from typing import Union

class ImagePreprocessor:
    # Constants for filter sizes and parameters
    GAUSSIAN_KERNEL_SIZE = (5, 5)
    GAUSSIAN_SIGMA = 0
    MEDIAN_KERNEL_SIZE = 5
    BILATERAL_DIAMETER = 9
    BILATERAL_SIGMA_COLOR = 75
    BILATERAL_SIGMA_SPACE = 75

    def __init__(self, image: Union[str, np.ndarray]):
        self.original_image = self._load_image(image)
        self.processed_image = self.original_image.copy()

    def _load_image(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """Load image from path or use provided numpy array"""
        if isinstance(image, str):
            return cv2.imread(image)
        return image

    def denoise(self, method: str = 'gaussian') -> 'ImagePreprocessor':
        """
        Apply denoising to the image using various methods.
        
        Args:
            method (str): Denoising method ('gaussian', 'median', 'bilateral', 'nlm')
        """
        if method == 'gaussian':
            self.processed_image = cv2.GaussianBlur(
                self.processed_image, 
                self.GAUSSIAN_KERNEL_SIZE, 
                self.GAUSSIAN_SIGMA
            )
        elif method == 'median':
            self.processed_image = cv2.medianBlur(
                self.processed_image, 
                self.MEDIAN_KERNEL_SIZE
            )
        elif method == 'bilateral':
            self.processed_image = cv2.bilateralFilter(
                self.processed_image, 
                self.BILATERAL_DIAMETER,
                self.BILATERAL_SIGMA_COLOR, 
                self.BILATERAL_SIGMA_SPACE
            )
        elif method == 'nlm':
            self.processed_image = (
                cv2.fastNlMeansDenoisingColored(self.processed_image)
                if len(self.processed_image.shape) == 3
                else cv2.fastNlMeansDenoising(self.processed_image)
            )
        else:
            raise ValueError(f"Unknown denoising method: {method}")
        return self

    def to_grayscale(self) -> 'ImagePreprocessor':
        """Convert image to grayscale if needed"""
        if len(self.processed_image.shape) == 3:
            self.processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
        return self

    def apply_threshold(self) -> 'ImagePreprocessor':
        """Apply Otsu's thresholding"""
        _, self.processed_image = cv2.threshold(
            self.processed_image, 0, 255, 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return self

    def get_result(self) -> np.ndarray:
        """Get the processed image"""
        return self.processed_image

def preprocess_for_ocr(image: Union[str, np.ndarray]) -> np.ndarray:
    """
    Preprocess image for better OCR results.
    
    Args:
        image (Union[str, np.ndarray]): Input image path or numpy array
    
    Returns:
        np.ndarray: Preprocessed image
    """
    preprocessor = (ImagePreprocessor(image)
                   .to_grayscale()
                   .denoise(method='gaussian')
                   .apply_threshold())
    return preprocessor.get_result()