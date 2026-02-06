import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "google/flan-t5-base"
DATA_PATH = "training/train.jsonl"
OUTPUT_DIR = "lora_model"

dataset = load_dataset(
    "json",
    data_files=DATA_PATH,
    split="train"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(example):
    """
    Convert raw QA into instruction format
    """
    prompt = (
        "Answer the question using company policy only.\n\n"
        f"Question: {example['question']}\n"
        "Answer:"
    )

    model_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=256
    )

    labels = tokenizer(
        example["answer"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=5,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=False,             
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("LoRA fine-tuning completed successfully")
