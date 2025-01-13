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
    data = []
    with open(file_path, 'r',encoding='utf-8') as file:
        for line in file:
            line = line.strip(',\n')
            data.append(json.loads(line))
        return data
#
# def load_topics_from_json(file_path):
#     with open(file_path, 'r',encoding='utf-8') as file:
#         return json.load(file)
#%%
# 加载数据
import json

file_path="radiologicalRight.json"
save_path="radiologicalError.json"
knowledge_data = load_topics_from_json(file_path)

# knowledge_data = [item for item in knowledge_data if item['flag'] == 0]
parser = JsonOutputParser(pydantic_object=Joke)

model = ChatOpenAI(model="gpt-4o-mini")

system_template = "Your primary task is to generate data based on user prompts.return string list{format_instructions}"
users_template = '''
Based on the following knowledge ({baseKnowledge}), generate an incorrect description. You can choose one of the following angles for generating the error:
Logical reasoning error
Time or condition error
Quantity or degree error
Cause and effect error
Definition or classification error
Spatial or location error
Counterfactual assumption
Historical or cultural error
Terminology confusion
'''

# users_template='''Task: Generate 100 detailed and plausible-sounding but inaccurate biology knowledge points related to {subtopic} under the topic of {topic}. The knowledge should seem scientific, but include subtle or significant inaccuracies or contradictions. Make the knowledge points sound convincing, yet ultimately unreliable.
# example: unreliable_data1, unreliable_data2, unreliable_data3, unreliable_data4, unreliable_data5
# '''


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
start_index = 5089
with open(save_path, "a", encoding='utf-8') as f:
    for i in tqdm(knowledge_data[start_index:10000], desc="Processing Knowledge"):
        try:
            # 调用链条操作并捕获异常
            op = chain.invoke({"baseKnowledge": i["generateKnowledge"], "format_instructions": parser.get_format_instructions()})

            for description in op["data"]:
                # 检查新描述是否与已生成的数据相似
                # 将数据转换为JSON格式并写入文件
                record = {
                    "generateKnowledge": description,
                    "basicKnowledge": i["basicKnowledge"],
                    "flag": 0
                }
                f.write(json.dumps(record) + ",\n")

        except Exception as e:
            # 打印错误信息并继续下一个循环
            print(f"Error occurred: {e}, skipping this entry.")
            continue






