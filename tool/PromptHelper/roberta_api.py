import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification


class ContradictionInferencer:
    def __init__(self, contradict_model_path='D:/PycharmProjects/TDFilter/textclassific/contradiction_biological_roberta_trueEnviroment_28/checkpoint-500'):
        # 加载训练好的模型和tokenizer
        self.contradict_model = RobertaForSequenceClassification.from_pretrained(contradict_model_path)
        self.contradict_tokenizer = RobertaTokenizer.from_pretrained(contradict_model_path)
        self.contradict_model.eval()  # 设置为评估模式

    def infer(self, basic_knowledge_list, generate_knowledge_list):
        # 确保输入长度一致
        assert len(basic_knowledge_list) == len(generate_knowledge_list), "输入列表长度不一致"

        # 编码输入
        inputs = self.contradict_tokenizer(
            basic_knowledge_list,
            generate_knowledge_list,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        # 禁用梯度计算
        with torch.no_grad():
            outputs = self.contradict_model(**inputs)
            logits = outputs.logits

        # 获取预测类别
        predicted_classes = logits.argmax(-1).tolist()  # 转换为列表
        return predicted_classes


import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification


class ConfidenceInferencer:
    def __init__(self, model_path='D:/PycharmProjects/TDFilter/textclassific/Biological_roberta_10_5/checkpoint-280'):
        # 加载训练好的模型和tokenizer
        self.model = RobertaForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model.eval()  # 设置为评估模式

    def infer(self, statements):
        # 编码输入
        inputs = self.tokenizer(
            statements,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        # 禁用梯度计算
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # 获取预测类别
        probabilities = torch.softmax(logits, dim=-1)

        # 获取每个预测的类别和对应的概率（可信度）
        batch_predictions = torch.argmax(logits, dim=-1).tolist()
        batch_confidences = [prob[cls].item() for prob, cls in zip(probabilities, batch_predictions)]

        return batch_predictions, batch_confidences


# 示例使用
if __name__ == "__main__":
    # 创建推理器实例
    trustworthiness_classifier = ConfidenceInferencer()

    # 输入示例
    statements = [
        "The Earth is flat.",
        "In some species, like peafowl, the female's choice of mate can significantly influence the male's reproductive success through selective pressure.",
        "Vaccines cause autism.",
        "Substance P is involved in the body's stress response and anxiety regulation.",
    ]

    # 调用推理函数
    results, probabilities = trustworthiness_classifier.infer(statements)

    # 输出结果
    for idx, (result, prob) in enumerate(zip(results, probabilities)):
        print(f"陈述 {idx + 1}: {'可信' if result == 1 else '不可信'}, 概率: {prob}")

# 示例使用
if __name__ == "__main__":
    # 创建推理器实例
    inferencer = ContradictionInferencer()

    # 输入示例
    basic_knowledge_list = [
        "In certain species, such as the blue-footed booby, elaborate courtship displays play a crucial role in mate selection.",
        "1+1=2",
        "Substance P is involved in the body's stress response and anxiety regulation."
    ]
    generate_knowledge_list = [
        "In many bird species, vibrant plumage and intricate songs are also important factors in attracting mates.",
        "1+1=3",
        "Substance P is involved in the body's stress response"
    ]



    # 调用推理函数
    results = inferencer.infer(basic_knowledge_list, generate_knowledge_list)
    for idx, result in enumerate(results):
        print(f"预测结果 {idx + 1}: {'不矛盾' if result == 1 else '矛盾'}")
