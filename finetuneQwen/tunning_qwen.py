import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import json
# Load tokenizer and model
model_name = "Qwen/Qwen-2"  # Replace with the exact model name if different
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
training_path = "D:/PycharmProjects/TDFilter/tool/dataGeneration/train_confidence.json"
# Load dataset
data = [
    {
        "input": "Based on your prior knowledge in the field of biology, evaluate the following description...",
        "output": "{\"result\": 0.9, \"explanation\": \"The validation information is valid because...\"}"
    },
    # Add more training examples
]
dataset = load_dataset('json', data_files={'train': training_path})


# Preprocess dataset
def preprocess_function(examples):
    inputs = [ex['input'] for ex in examples['train']]
    outputs = [ex['output'] for ex in examples['train']]
    model_inputs = tokenizer(inputs, max_length=2048, padding="max_length", truncation=True)
    labels = tokenizer(outputs, max_length=1024, padding="max_length", truncation=True)

    # Shift the labels for causal LM
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "v_proj"]  # Qwen2 uses multi-head attention; adjust target layers
)

# Apply LoRA to the model
peft_model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./qwen2_lora_biology",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
)

# Trainer
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()

# Save the model
peft_model.save_pretrained("./qwen2_lora_biology")

# Evaluate the model or use it for inference
