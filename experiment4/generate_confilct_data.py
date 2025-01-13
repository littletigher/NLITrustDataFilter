import os
import json
from langchain import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from tqdm import tqdm

# 设置API key和API base URL

os.environ["OPENAI_API_KEY"] = "sk-R7oU7otOgPSU7KU6666070C271574aA085FdE47a74FdD92a"
os.environ["OPENAI_BASE_URL"] = "https://api.bianxie.ai/v1"

class Joke(BaseModel):
    data: list = Field(description="result of generate data json")

parser = JsonOutputParser(pydantic_object=Joke)

# 定义模型
model = ChatOpenAI(model="gpt-4o-mini")
# 定义任务模板
system_template = "You are a helpful assistant."
user_template = '''
Based on the following knowledge:
Topic: {topic}
Subtopic: {subtopic}
Knowledge: "{knowledge}"

Please generate two new pieces of information:
1. One that is logically consistent with the original knowledge (flag 1).
2. One that contradicts the original knowledge (flag 0).

Provide the output in the format:
[
    {{
        "newGenerateKnowledge": "",
        "flag": 1,
        "explanation": "The validation information is not contradictory because ..."
    }},
    {{
        "newGenerateKnowledge": "",
        "flag": 0,
        "explanation": "The validation information is contradictory because ..."
    }}
]
'''

# 创建LangChain的Prompt模板
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_template),
        ("user", user_template)
    ]
)

# 使用LangChain构建一个LLM链

chain = prompt_template | model | parser
# 读取进度
def load_progress(index_file):
    try:
        with open(index_file, 'r', encoding='utf-8') as f:
            progress = json.load(f)
            return progress.get("last_processed_index", -1)
    except FileNotFoundError:
        return -1


# 保存进度
def save_progress(index, index_file):
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump({"last_processed_index": index}, f)

def generate_data_for_topic(topic, subtopic, knowledge):
    try:
        # 生成数据
        result = chain.invoke({
            "topic": topic,
            "subtopic": subtopic,
            "knowledge": knowledge
        })
        return result  # 将结果转换为JSON格式
    except Exception as e:
        print(f"Error during data generation: {e}")
        return []


# 保存单个结果到 JSON 文件，使用 'a' 模式追加写入
def save_single_result(result, file_path):
    json_result = json.dumps(result, ensure_ascii=False, indent=4)
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json_result + '\n')


# 主处理函数
def process_topic_data(topic_entry, index, output_file):
    topic = topic_entry["topic"]
    subtopic = topic_entry["subtopic"]
    generated_data = []
    # 读取子主题的知识点
    # 生成与知识点相关的两类数据
    new_data = generate_data_for_topic(topic, subtopic, topic_entry["generateKnowledge"])

        # 检查生成的记录
    for description in new_data:
        record = {
            "basicKnoledge": topic_entry["generateKnowledge"],
            "generateKnowledge": description["newGenerateKnowledge"],
            "topic": topic,
            "subtopic": subtopic,
            "flag": description["flag"],
            "explanation": description["explanation"]
        }
        generated_data.append(record)

    # 保存数据
    save_generated_data(generated_data, output_file)

    # 更新进度
    save_progress(index, "trash/index.json")


# 读取 topics_data 文件
def load_topics_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# 保存生成的数据到文件
def save_generated_data(generated_data, output_file):
    with open(output_file, "a", encoding='utf-8') as f:
        for record in generated_data:
            f.write(json.dumps(record) + ",\n")


# 处理所有主题数据
def process_all_topics(topics_data, output_file):
    # 加载进度
    last_processed_index = load_progress("trash/index.json")

    # 单线程顺序处理每个主题
    for i, topic_entry in enumerate(tqdm(topics_data, total=len(topics_data))):
        if i > last_processed_index:
            process_topic_data(topic_entry, i, output_file)


# 主函数
if __name__ == "__main__":
    # 从文件中加载 topics_data
    topics_data = load_topics_from_json("trash/experiment_data.json")

    # 开始处理
    process_all_topics(topics_data, "generate_confilct_data.json")
