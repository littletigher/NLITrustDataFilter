from transformers import pipeline


class ContradictionInferencer:
    def __init__(self):
        # 加载零样本分类模型
        self.classifier = pipeline("text-classification", model="FacebookAI/roberta-large-mnli",device=0)

    def infer(self, basic_knowledge_list, generate_knowledge_list):
        # 确保输入长度一致
        assert len(basic_knowledge_list) == len(generate_knowledge_list), "输入列表长度不一致"

        predicted_classes = []
        score_of_predict = []
        for premise, hypothesis in zip(basic_knowledge_list, generate_knowledge_list):
            # 构造输入文本
            input_text = f"premise: {premise} </s></s> hypothesis: {hypothesis}"
            # 获取预测结果
            result = self.classifier(input_text)
            # 判断预测标签
            label = result[0]['label'].lower()  # 获取预测标签并转换为小写
            if label == 'contradiction':
                predicted_classes.append(0)  # 矛盾
            elif label == 'entailment':
                predicted_classes.append(1)  # 不矛盾
            else:
                predicted_classes.append(-1)  # 中性
            score_of_predict.append(result[0]['score'])

        return predicted_classes,score_of_predict


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
    print(results)
    # for idx, result in enumerate(results):
    #     print(f"预测结果 {idx + 1}: {'不矛盾' if result == 1 else '矛盾' if result == 0 else '中性'}")
