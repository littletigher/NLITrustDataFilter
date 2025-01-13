import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, TrainingArguments, AutoTokenizer, Trainer, DataCollatorForSeq2Seq
from peft import get_peft_model, LoraConfig, TaskType

def count_parameters(model):
    """
    计算模型的可训练参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 定义tokenization函数
def preprocess_function(examples):
    # 只使用 instruction 作为输入
    inputs = examples['input']
    outputs_ = examples['output']

    MAX_LENGTH = 100000  # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性 这里不能做任何切割 否则训练效果无法达到
    instruction = tokenizer(
        f"<|im_start|>{inputs}<|im_end|>",
        add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(
        f"<|im_start|>{outputs_}<|im_end|>",
        add_special_tokens=False
    )
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


if __name__ == '__main__':
    # model_path = "/home/u2010813054/TDF/Qwen2-7B-Instruct"
    tokenizer_path = "E:\GitHub\model\Qwen2-7B-Instruct"
    source_data_path = "D:/PycharmProjects/TDFilter/tool/dataGeneration/train_confidence.json"
    # model_save_path = "qwen2_confidenct_lora"

    # 加载模型的Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,trust_remote_code=True)
    print("模型加载完毕")
    train_df = pd.read_json(source_data_path)
    print(train_df)
    # train_ds = Dataset.from_pandas(train_df)
    # print("数据集加载完毕")
    # train_dataset = train_ds.map(preprocess_function, remove_columns=train_ds.column_names)
    # print("数据集格式化结束")
    #
    # LoRA 配置，具体可以参考论文
    # peft_config = LoraConfig(
    #     task_type=TaskType.CAUSAL_LM,
    #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    #     inference_mode=False,  # 训练模式
    #     r=128,  # Lora 秩
    #     lora_alpha=256,  # Lora alaph，具体作用参见 Lora 原理
    # )
    #
    # model = AutoModelForCausalLM.from_pretrained(model_path, use_cache=False)
    #
    # pre_param = count_parameters(model)
    #
    # model = get_peft_model(model, peft_config)
    #
    # lora_param = count_parameters(model)
    # print("待微调模型加载完毕")
    # 训练参数
    # training_args = TrainingArguments(
    #     output_dir=model_save_path,
    #     per_device_train_batch_size=4,
    #     gradient_accumulation_steps=8,
    #     logging_steps=1,
    #     logging_dir='logs',
    #     num_train_epochs=5,
    #     save_steps=1000,
    #     learning_rate=3e-4,
    #     eval_strategy="no",
    #     bf16=True,
    #     save_total_limit=10,
    #     warmup_ratio=0.01,
    #     per_device_eval_batch_size=1,
    #     weight_decay=0.0,
    #     adam_beta2=0.95,
    #     lr_scheduler_type="cosine",
    #     report_to="none",
    # )
    #
    # 初始化Trainer
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    #     tokenizer=tokenizer
    # )
    #
    # 开始训练
    # print("开始训练")
    # trainer.train()
    #
    # 保存微调后的模型
    # print("保存微调后的模型")
    #
    # 保存模型权重和分词器
    # model.save_pretrained(model_save_path)
    # tokenizer.save_pretrained(model_save_path)
    #
    # print(f'微调前模型参数：{pre_param}，lora微调矩阵大小：{lora_param}')