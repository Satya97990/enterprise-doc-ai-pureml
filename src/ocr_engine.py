# src/ocr_engine.py
import cv2
import numpy as np
import pytesseract
from PIL import Image
from typing import Dict, List, Tuple, Union

class OCREngine:
    def __init__(self, lang: str = 'eng'):
        """
        Initializes the OCR Engine for document feature extraction.
        """
        self.lang = lang

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Converts PIL Image to OpenCV format, applies grayscale, and thresholding
        to improve Tesseract OCR accuracy on scanned documents.
        """
        # Convert PIL to cv2 (RGB to BGR)
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding to handle varying illumination
        processed = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return processed

    def normalize_bbox(self, bbox: Tuple[int, int, int, int], width: int, height: int) -> List[int]:
        """
        Normalizes a bounding box (x0, y0, x1, y1) to the 0-1000 scale required by LayoutLM.
        """
        x0, y0, x1, y1 = bbox
        return [
            int(1000 * (x0 / width)),
            int(1000 * (y0 / height)),
            int(1000 * (x1 / width)),
            int(1000 * (y1 / height))
        ]

    def extract_layout(self, image: Image.Image) -> Dict[str, Union[List[str], List[List[int]]]]:
        """
        Extracts words and their normalized bounding boxes from an image.
        
        Returns:
            Dictionary containing 'words' and 'bboxes' (normalized).
        """
        processed_img = self.preprocess_image(image)
        width, height = image.size
        
        # Extract data using pytesseract
        ocr_data = pytesseract.image_to_data(processed_img, lang=self.lang, output_type=pytesseract.Output.DICT)
        
        words = []
        bboxes = []
        
        for i in range(len(ocr_data['text'])):
            word = ocr_data['text'][i].strip()
            # Filter out empty strings and low-confidence predictions
            if word and int(ocr_data['conf'][i]) > 10:
                x = ocr_data['left'][i]
                y = ocr_data['top'][i]
                w = ocr_data['width'][i]
                h = ocr_data['height'][i]
                
                # LayoutLM expects [x0, y0, x1, y1]
                raw_bbox = (x, y, x + w, y + h)
                normalized_bbox = self.normalize_bbox(raw_bbox, width, height)
                
                # Clamp values to ensure they stay strictly within 0-1000
                clamped_bbox = [
                    max(0, min(1000, normalized_bbox[0])),
                    max(0, min(1000, normalized_bbox[1])),
                    max(0, min(1000, normalized_bbox[2])),
                    max(0, min(1000, normalized_bbox[3]))
                ]
                
                words.append(word)
                bboxes.append(clamped_bbox)
                
        return {
            "words": words,
            "bboxes": bboxes
        }

if __name__ == "__main__":
    # Quick module test
    engine = OCREngine()
    print("OCR Engine initialized successfully. Ready for preprocessing.")