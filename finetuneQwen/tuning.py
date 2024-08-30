from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import torch

def preprocess_function(examples):
    inputs_ = examples['input']
    outputs_ = examples['output']
    model_inputs = tokenizer(inputs_, max_length=1024, truncation=True, padding="max_length")
    labels = tokenizer(outputs_, max_length=1024, truncation=True, padding="max_length").input_ids
    labels_with_ignore_index = [-100 if token == tokenizer.pad_token_id else token for token in labels]
    model_inputs["labels"] = labels_with_ignore_index
    return model_inputs

if __name__ == '__main__':
    model_path = "model/Qwen2-7B-Instruct"
    tokenizer_path = "model/Qwen2-7B-Instruct"
    source_data_path = "data_source/chemical_warfare_fineturning_data.jsonl"
    model_save_path = "target/"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    dataset = load_dataset("json", data_files=source_data_path)
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1
    )

    model = AutoModelForCausalLM.from_pretrained(model_path)
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir="logs",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        save_steps=10_000,
        logging_steps=500,
        push_to_hub=False,
        fp16=True,
        dataloader_num_workers=4,
        deepspeed=deepspeed_config,  # 使用DeepSpeed
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
    )

    trainer.train()
    model.save_pretrained(model_save_path+"lora_qwen2-7b")
    tokenizer.save_pretrained(model_save_path+"lora_qwen2-7b")

    input_text = "What is chemical warfare?"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=150)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
