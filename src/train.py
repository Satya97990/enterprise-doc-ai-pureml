# src/train.py
import os
import mlflow
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)
from src.dataset import FUNSDDataset, get_dataloader_labels
from src.evaluate import get_compute_metrics_fn

def main():
    # 1. Initialize MLflow tracking for experiment visualization
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "layoutlmv3-funsd-ner"
    mlflow.set_tracking_uri("sqlite:///mlruns.db")

    # 2. Setup label mappings
    id2label, label2id = get_dataloader_labels()
    num_labels = len(id2label)

    # 3. Load processor and unified modalities
    print("Loading LayoutLMv3 processor and datasets...")
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    
    train_dataset = FUNSDDataset(processor, split="train")
    eval_dataset = FUNSDDataset(processor, split="test")

    # 4. Initialize model architecture
    print("Initializing LayoutLMv3ForTokenClassification...")
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base",
        id2label=id2label,
        label2id=label2id,
        num_labels=num_labels
    )

    # 5. Define Training Arguments
    # Training optimized with a learning rate of 5e-5
    training_args = TrainingArguments(
        output_dir="models/layoutlmv3_finetuned",
        max_steps=1000,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        weight_decay=0.01,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="mlflow",
        # Critical: LayoutLMv3 forwards image pixel_values, so standard text column dropping must be disabled
        remove_unused_columns=False, 
    )

    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor,
        data_collator=DefaultDataCollator(),
        compute_metrics=get_compute_metrics_fn(id2label)
    )

    # 7. Execute Training Loop
    print("Starting fine-tuning...")
    trainer.train()

    # 8. Evaluate and Save
    print("Evaluating best model...")
    metrics = trainer.evaluate()
    print(f"Final Evaluation Metrics: {metrics}")

    print("Saving fine-tuned model and processor...")
    trainer.save_model("models/layoutlmv3_finetuned")
    processor.save_pretrained("models/tokenizer")

if __name__ == "__main__":
    main()