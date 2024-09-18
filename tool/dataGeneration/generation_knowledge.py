import openai
import json
import time
# 使用prompt 提示词进行生成
from tqdm import tqdm
from tool.PromptHelper.gpt_api import api_generation
# 从 JSON 文件加载主题和子主题
def load_topics_from_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


# 定义用于生成知识点的函数
def generate_knowledge(topic, subtopic, max_retries=3):
    prompt = f"Generate 5 detailed and accurate biology knowledge points related to {subtopic} under the topic of {topic}. The knowledge should be concise, informative, and factual."

    retries = 0
    while retries < max_retries:
        try:
            response = api_generation([prompt])
            return response[0].choices[0].text.strip()
        except Exception as e:
            retries += 1
            print(f"Error generating knowledge for {subtopic} under {topic}: {e}")
            time.sleep(2)  # 等待2秒后重试
    return None


# 保存生成的知识到文件
def save_knowledge_to_file(knowledge_data, output_file):
    with open(output_file, 'a') as file:
        json.dump(knowledge_data, file, indent=4)
        file.write('\n')


# 主函数
def main(input_json, output_json):
    # 加载主题和子主题
    topics_data = load_topics_from_json(input_json)
    knowledge_data = []

    # 计算总的子主题数量，用于进度条的总长度
    total_subtopics = sum(len(item["subtopics"]) for item in topics_data)

    # 创建进度条
    with tqdm(total=total_subtopics*40) as pbar:
        # 遍历每个主题和子主题
        for item in topics_data:
            topic = item["topic"]
            for subtopic in item["subtopics"]:
                for i in range(0,40):
                    print(f"Generating knowledge for {topic} - {subtopic}...")
                    knowledge_points = generate_knowledge(topic, subtopic)

                    if knowledge_points:
                        knowledge_entry = {
                            "topic": topic,
                            "subtopic": subtopic,
                            "knowledge": knowledge_points.split('\n')  # 分割成多个知识点
                        }
                        knowledge_data.append(knowledge_entry)
                        save_knowledge_to_file(knowledge_entry, output_json)
                    # 保存生成的知识到文件
                    time.sleep(0.2)  # 防止API速率限制

                    # 更新进度条
                    pbar.update(1)

    # 将生成的知识保存到文件

# 运行代码
if __name__ == "__main__":
    input_json = "D:/PycharmProjects/TDFilter/data/generateData/biological__knoledge_category.jsonl"  # 包含主题和子主题的JSON文件
    output_json = "D:/PycharmProjects/TDFilter/tool/dataGeneration/generate.json"  # 输出生成的知识的JSON文件
    main(input_json, output_json)
