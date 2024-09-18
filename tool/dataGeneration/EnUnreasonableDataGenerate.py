#使用gpt4 尽量生成质量较高的不合理数据
#进行去重相似度匹配操作
#使用进程池进行并行处理
# template for generating unreasonable data

#标号为 n 的数据使用 n/10+1 作为category   n%10 作为example


import getpass
import os
import json
import pandas as pd
from tqdm import tqdm
from langchain_core.pydantic_v1 import BaseModel, Field
os.environ["OPENAI_API_KEY"] = "sk-R7oU7otOgPSU7KU6666070C271574aA085FdE47a74FdD92a"
os.environ["OPENAI_BASE_URL"] = "https://api.bianxie.ai/v1"

from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.messages import HumanMessage, SystemMessage

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

class Joke(BaseModel):
    data: list = Field(description="result of generate data string")
def load_topics_from_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

topics_data = load_topics_from_json("/data/generateData/catalog/biological__knoledge_category.jsonl")

parser = JsonOutputParser(pydantic_object=Joke)

model = ChatOpenAI(model="gpt-4o-mini")

system_template = "Your primary task is to generate data based on user prompts.return string list{format_instructions}"
# users_template = '''
# Task: Generate 100 detailed and accurate biology knowledge points related to {subtopic} under the topic of {topic}. The knowledge should be concise, informative, and factual.
# example:generated_data1,generated_data2,generated_data3,generated_data4,generated_data5
# '''

users_template='''Task: Generate 100 detailed and plausible-sounding but inaccurate biology knowledge points related to {subtopic} under the topic of {topic}. The knowledge should seem scientific, but include subtle or significant inaccuracies or contradictions. Make the knowledge points sound convincing, yet ultimately unreliable.
example: unreliable_data1, unreliable_data2, unreliable_data3, unreliable_data4, unreliable_data5
'''


#—————————————— 上方参数需要在之后进行工程化处理 ——————————————




prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", users_template)]
)

chain = prompt_template | model | parser

# 暂时使用三重for循环构建数据生成基础框架

file_index = 1
data_count = 0
generated_data = []
# 假设 categories_and_examples 是一个列表，你可以根据实际情况进行调整
init_index =4
with open("D:/PycharmProjects/TDFilter/tool/dataGeneration/generate_incorrect.json", "a", encoding='utf-8') as f:
    for i in tqdm(topics_data, desc="Processing topic"):
        for j in tqdm(i["subtopics"], desc="Processing subtopics"):
            # for k in tqdm(range(10), desc="Generating Data", leave=False):
            op = chain.invoke({"topic": i["topic"],
                            "subtopic": j,
                            "format_instructions": parser.get_format_instructions()})
            # 将数据转换为JSON格式并写入文件
            for description in op["data"]:
                # 检查新描述是否与已生成的数据相似
                # 将数据转换为JSON格式并写入文件
                record = {
                    "generateKnowledge": description,
                    "topic": i["topic"],
                    "subtopic": j,
                    "flag": 1
                }
                f.write(json.dumps(record) + ",\n")








