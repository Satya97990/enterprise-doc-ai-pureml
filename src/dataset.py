# src/dataset.py
import torch
from torch.utils.data import Dataset
from transformers import LayoutLMv3Processor
from datasets import load_dataset
from typing import Dict, Any

# FUNSD standard labels
LABEL_LIST = [
    "O",
    "B-HEADER",
    "I-HEADER",
    "B-QUESTION",
    "I-QUESTION",
    "B-ANSWER",
    "I-ANSWER"
]

class FUNSDDataset(Dataset):
    def __init__(self, processor: LayoutLMv3Processor, split: str = "train"):
        """
        Initializes the PyTorch dataset mapping FUNSD data to LayoutLMv3 tensors.
        
        Args:
            processor (LayoutLMv3Processor): Unified Hugging Face processor (Feature Extractor + Tokenizer).
            split (str): 'train' or 'test'.
        """
        self.processor = processor
        # Load the FUNSD dataset directly from Hugging Face
        self.dataset = load_dataset("nielsr/funsd-layoutlmv3", split=split)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves and processes a single document image and its annotations.
        """
        item = self.dataset[idx]
        image = item["image"]
        words = item["tokens"]
        boxes = item["bboxes"]
        word_labels = item["ner_tags"]

        # The LayoutLMv3Processor handles:
        # 1. Resizing and normalizing the image for the Vision Transformer.
        # 2. Tokenizing words and applying WordPiece sub-tokenization.
        # 3. Expanding bounding boxes to match sub-tokens.
        # 4. Aligning labels with sub-tokens (setting -100 for sub-words to ignore in loss computation).
        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            word_labels=word_labels,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )

        # The processor returns batch-level tensors (shape: [1, seq_len]). 
        # We squeeze the first dimension to return item-level tensors.
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "bbox": encoding["bbox"].squeeze(),
            "labels": encoding["labels"].squeeze(),
            "pixel_values": encoding["pixel_values"].squeeze()
        }

def get_dataloader_labels():
    """Returns the mapping of integer IDs to string tags for evaluation."""
    id2label = {id: label for id, label in enumerate(LABEL_LIST)}
    label2id = {label: id for id, label in enumerate(LABEL_LIST)}
    return id2label, label2id

if __name__ == "__main__":
    # Validate the dataset processing pipeline
    try:
        print("Initializing processor...")
        processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        print("Loading FUNSD training split...")
        train_dataset = FUNSDDataset(processor, split="train")
        sample = train_dataset[0]
        
        print("\nSuccessfully loaded sample 0.")
        print(f"Input IDs shape: {sample['input_ids'].shape}")
        print(f"Bounding Boxes shape: {sample['bbox'].shape}")
        print(f"Pixel Values shape: {sample['pixel_values'].shape}")
        print(f"Labels shape: {sample['labels'].shape}")
    except Exception as e:
        print(f"Error initializing dataset: {e}")