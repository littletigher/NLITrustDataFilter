# 使用迭代的方法对模型进行过滤
import random

from tool.Filter.Contradiction_Filter import ContradictionFilter
from tool.Filter.Reasonable_Filter import ReasonableFilter
from tool.LangchainHelper.langchainMilvusHelper import LangchainMilvusHelper
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv
from tool.Milvus.MilvusWrapper import MilvusWrapper
from pymilvus import connections, Collection
from tool.PromptHelper.chatgpt3_5 import api_generation
from tool.DataProcess.embed import embed_text
import numpy as np
import json
import os
import re

# 加载环境变量
def get_config():
    """
    Gets configuration from environment variables.

    Returns:
    - A dictionary with configuration parameters.
    """
    try:
        load_dotenv()
        config_ = {
            "output_file": os.getenv("results_file"),
            "checkpoint_file": os.getenv("checkpoint_file"),
            "save_data_path": os.getenv("save_data_path"),
            "unresolved_file": os.getenv("unresolved_file"),
            "iterater_file": os.getenv("iterater_file"),
            # "instruction_data_file": os.getenv("INSTRUCTION_DATA_FILE"),
            "request_batch_size": int(os.getenv("FINE_TUNE_DATA_BATCH_SIZE")),
            "validate_data_path": os.getenv("validate_data_path"),
        }
        return config_
    except ValueError as e:
        print(f"环境变量配置错误: {e}")
        exit(1)

class IterateFilterProcess:
    '''
    迭代过滤器
    每次迭代过滤10%的数据，并将正确数据存储到数据库中
    第一次编码 这10%的数据 为随机选取的数据
    '''
    def __init__(self):
        # self.milvusWrapperDatasets = MilvusWrapper()
        self.milvusWrapperKnowledge = MilvusWrapper(host='10.129.205.160', port='19530', collection_name='basic_filter_exp4_gpt')

    # 处理输入的数据 这里统一假设数据为json格式
    # 包含两个字段 一个是id 一个是文本
    # 他是json格式的数据
    import json

    def process_json_data(self, filename):
        '''
        验证通过
        处理json格式的数据
        :param filename: JSON文件的路径
        :return: 解析后的数据或异常
        '''

        try:
            # 读取文件内容
            with open(filename, 'r', encoding='utf-8') as file:
                data = file.read()

            # 假设数据为 JSON 格式，进行解析
            parsed_data = json.loads(data)

            # 检查数据中是否包含 'id' 和 'text' 字段
            return parsed_data
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format")

        except FileNotFoundError:
            raise ValueError(f"File {filename} not found")

    def process_milvus_data(self, collection_name):
        '''
        从milvus中获取数据
        :param collection_name:
        :return:
        '''
        self.milvusWrapperDatasets.collection = Collection(collection_name)
        self.validate_dataset=self.milvusWrapperDatasets.collection.query(expr="id >= 0", output_fields=["id","knowledge" ,"flag"])
        return self.validate_dataset
    # 提示词构建 校验是否矛盾
    def get_conflict_prompts(self, valided_datas) -> list:
        '''
        提示词构建，校验矛盾与否
        :param valided_datas:
        :return:
        '''
        prompts_ = []
        ids_ = []
        for validata in valided_datas:

            matching_knowledge = self.get_confidence_data(validata,mod="local")

            if not matching_knowledge:
                prompts_.append("no matching knowledge")
                ids_.append(-1)
                continue
            # if not matching_knowledge:
            #     continue
            id=matching_knowledge[0]['id']
            prompt = (
                    f'''Based on the provided known information: {matching_knowledge[0]['knowledge']}, evaluate whether the validation information: {validata} is logically consistent or contradictory.
            Use the known information to determine if the validation information can be inferred or contradicted. If the known information is not closely related to the validation information, assume they are unrelated.
            Provide a result in one of the following categories:
            1 indicates the validation information is not contradictory to the known information,
            0 indicates the validation information contradicts the known information,
            -1 indicates the validation information is unrelated to the known information.

            Provide an explanation for your decision in the following JSON format:
            {{"result": 1/0/-1, "explanation": "The validation information is [not contradictory/contradictory/unrelated] because .."}}'''

            )
            prompts_.append(prompt)
            ids_.append(id)
        return prompts_,ids_

    def get_confidence_data(self, validate_data:str, top_k=1,mod="local"):
        if(mod=="local"):
            return self.milvusWrapperKnowledge.query(validate_data, top_k=top_k)
        else:
            return None

    # 对数据进行处理
    def get_reasonable_prompts(self, valided_datas) -> list:
        '''
        提示词构建，校验合理与否
        :param valided_datas:
        :return:
        '''
        prompts_ = []
        for validata in valided_datas:
            # matching_knowledge = self.get_confidence_data(validata)
            # if not matching_knowledge:
            #     continue

            prompt = (
                        f'''Based on your prior knowledge in the field of biology, evaluate the following description and determine if it is reasonable. Description: {validata}. Provide a result in one of the following categories: 
                        1 indicates the information is valid and trustworthy, 
                        0 indicates the information is invalid or not trustworthy, 
                        -1 indicates that it is impossible to determine the validity of the information based on the available prior knowledge. 
                        Return your decision and reasoning in the following JSON format: 
                        {{\"result\": 1/0/-1, \"explanation\": \"The validation information is [correct/incorrect/undeterminable] because ...\"}}'''

            )
            prompts_.append(prompt)
        return prompts_
    def pre_install(self,datasets):
        knowledge_list = [item["generateKnowledge"] for item in datasets]
        flag_list = [item["flag"] for item in datasets]
        topic_list = [item["topic"] for item in datasets]
        for i in range(0,len(datasets)):
            embedding = embed_text(knowledge_list[i])  #
            embedding_np = embedding.astype(np.float32).tolist()
            save_data = [
                embedding_np,
                [knowledge_list[i]],
                [topic_list[i]],
                [flag_list[i]],
                [1],  # Confidenct
                [-1],  # Lambda
                [-1]  # Relation_id
            ]
            # insert_data([
            #                 embedding_np,  # embedding
            #                 [text],  # knowledge
            #                 [flag]])
            # 存储到数据库中
            self.milvusWrapperKnowledge.insert_data(save_data)

    def process_iterator(self, datasets) -> list:
        '''
        处理数据，多轮过滤
        :param datasets:
        :return:
        '''
        config = get_config()
        last_index = self.load_checkpoint(config["checkpoint_file"])

        # 用于暂存无法判断的记录
        unresolved_conflicts = self.load_unresolved_conflicts(config["unresolved_file"])

        request_batch_size = config["request_batch_size"]
        generate_count = len(datasets)
        # generate_count = 200
        # 处理数据集
        knowledge_list = [item["generateKnowledge"] for item in datasets]
        flag_list = [item["flag"] for item in datasets]

        topic_list = [item["topic"] for item in datasets]

        # 定义最大迭代次数，避免无限循环
        max_iterations = 3
        iteration = 0

        # 每次过滤时创建或追加文件
        with open(config["output_file"], "a", encoding='utf-8') as f:
            while iteration < max_iterations:
                iteration += 1
                print(f"开始第 {iteration} 次过滤...")

                for i in tqdm(range(last_index, generate_count, request_batch_size)):
                    conflict_prompts, conflict_ids = self.get_conflict_prompts(knowledge_list[i:i + request_batch_size])

                    reasonable_prompts = self.get_reasonable_prompts(knowledge_list[i:i + request_batch_size])
                    results_conflict = api_generation(conflict_prompts)
                    results_reasonable = api_generation(reasonable_prompts)

                    for j in range(len(conflict_prompts)):
                        index = i + j


                        result_conflict = results_conflict[j]
                        result_reasonable = results_reasonable[j]
                        response_conflict = result_conflict.get("response")
                        response_reasonable = result_reasonable.get("response")


                        try:
                            response_conflict = json.loads(response_conflict.replace('：', ':'))
                            response_reasonable = json.loads(response_reasonable.replace('：', ':'))
                        except:
                            unresolved_conflicts.append(datasets[index])
                            self.update_unresolved_conflicts(config["unresolved_file"], unresolved_conflicts)
                            print(f"无法判断记录: {index}")
                            continue

                        # 检查是否无法判断，response_conflict['result'] == -1
                        if response_conflict['result'] == -1:
                            unresolved_conflicts.append(datasets[index])
                            self.update_unresolved_conflicts(config["unresolved_file"], unresolved_conflicts)
                            print(f"无法判断记录: {index}")
                            continue  # 跳过该条，等待下次过滤
                         # 跳过该条，等待下次过滤
                        # 正常处理可判断的记录
                        record = {
                            "id": index,
                            "conflict_score": response_conflict['result'],
                            "reasonable_score": response_reasonable['result'],
                            "topic": topic_list[index],
                            "conflict_explanation": response_conflict['explanation'],
                            "reasonable_explanation": response_reasonable['explanation'],
                            "conflict_prompt": conflict_prompts[j],
                            "reasonable_prompt": reasonable_prompts[j],
                            "relation_id": conflict_ids[j]
                        }
                        embedding = embed_text(knowledge_list[i])
                        embedding_np = embedding.astype(np.float32).tolist()

                        if (response_conflict['result'] == 1 and response_reasonable['result']  == 1) or (response_conflict['result'] == -1 and response_reasonable['result']  == 1) or (response_conflict['result'] == 1 and response_reasonable['result']  == -1):
                            save_data = [
                                embedding_np,
                                [knowledge_list[index]],
                                [topic_list[index]],
                                [flag_list[index]],
                                [response_reasonable['result']],
                                [response_conflict['result']],
                                [conflict_ids[j]]
                            ]
                            self.milvusWrapperKnowledge.insert_data(save_data)

                        f.write(json.dumps(record, ensure_ascii=False) + '\n')
                        self.update_checkpoint(config["checkpoint_file"], index)

                # 如果没有未解决的冲突则结束
                if not unresolved_conflicts:
                    print("所有数据处理完毕，无需进一步过滤。")
                    break

                # 准备下一次过滤的内容
                print(f"第 {iteration} 轮过滤完毕，剩余未解决的记录: {len(unresolved_conflicts)}")
                datasets = unresolved_conflicts
                unresolved_conflicts = []
                last_index = 0
                self.update_checkpoint(config["checkpoint_file"], 0)
                with open(config["iterater_file"], 'w', encoding='utf-8') as uf:
                    json.dump(datasets, uf, ensure_ascii=False)

                generate_count=len(datasets)
                print(datasets)
                # 处理数据集
                knowledge_list = [item["generateKnowledge"] for item in datasets]
                flag_list = [item["flag"] for item in datasets]
                topic_list = [item["topic"] for item in datasets]

        # print("将最后剩下的无法判断的数据处理掉")
        self.process_iterator_oneTime(datasets)

    def process_iterator_oneTime(self,datasets) -> list:
        '''
        处理数据
        :param valided_datas:
        :return:
        '''

        config = get_config()
        last_index = self.load_checkpoint(config["checkpoint_file"])
        # 断点保存
        request_batch_size = config["request_batch_size"]
        generate_count =  len(datasets)
        # 分割获得各个list

        knowledge_list = [item["generateKnowledge"] for item in datasets]
        flag_list = [item["flag"] for item in datasets]
        # id_list = [item["id"] for item in datasets]
        topic_list = [item["topic"] for item in datasets]

        #迭代运行（注意 目前先不使用迭代，而是一边运行一边过滤）
        #save_count = 1400
        # 使用两种方法，迭代处理数据
        with open(config["output_file"], "a",encoding='utf-8') as f:
            for i in tqdm(range(last_index, generate_count, request_batch_size)):
                conflict_prompts,conflict_ids = self.get_conflict_prompts(knowledge_list[i:i + request_batch_size])
                reasonable_prompts = self.get_reasonable_prompts(knowledge_list[i:i + request_batch_size])
                results_conflict = api_generation(conflict_prompts)
                results_reasonable = api_generation(reasonable_prompts)
                for j in range(len(conflict_prompts)):
                    result_conflict = results_conflict[j]
                    result_reasonable = results_reasonable[j]
                    response_conflict = result_conflict.get("response")
                    response_reasonable = result_reasonable.get("response")
                    index = i + j
                    # 处理response字段，将其转换为包含input和output的字典
                    # output_lines = response.split("\n")
                    # input_value = output_lines[0].replace("input: ", "")
                    # output_value = output_lines[1].replace("output: ", "")
                    # if output_value == "" or input_value == "":
                    #    index -= 1
                    #    continue
                    try:
                        response_conflict = json.loads(response_conflict.replace('：', ':'))
                        response_reasonable = json.loads(response_reasonable.replace('：', ':'))
                    except:
                        continue
                    record = {
                        "id": index,
                        "flag": flag_list[index],
                        "conflict_score": response_conflict['result'],
                        "reasonable_score": response_reasonable['result'],
                        "topic": topic_list[index],
                        "conflict_explanation": response_conflict['explanation'],
                        "reasonable_explanation": response_reasonable['explanation'],
                        "conflict_prompt": conflict_prompts[j],
                        "reasonable_prompt": reasonable_prompts[j],
                        "relation_id":conflict_ids[j]
                    }
                    embedding = embed_text(knowledge_list[i])  # 获取嵌入向量

                    # 如果 embedding 是 NumPy 数组，直接转换为 list
                    embedding_np = embedding.astype(np.float32).tolist()
                    if (response_conflict['result'] == 1 and response_reasonable['result'] == 1) or (response_conflict['result'] == -1 and response_reasonable['result'] == 1) or (response_conflict['result'] == 1 and response_reasonable['result'] == -1):
                        save_data=[
                            embedding_np,
                            [knowledge_list[index]],
                            [topic_list[index]],
                            [flag_list[index]],
                            [response_reasonable['result']] , #Confidenct
                            [response_conflict['result']], #Lambda
                            [conflict_ids[j]] #Relation_id
                        ]
                        self.milvusWrapperKnowledge.insert_data(save_data)
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    self.update_checkpoint(config["checkpoint_file"], index)

    def process(self,datasets) -> list:
        '''
        处理数据
        :param valided_datas:
        :return:
        '''

        config = get_config()
        last_index = self.load_checkpoint(config["checkpoint_file"])
        # 断点保存
        request_batch_size = config["request_batch_size"]
        generate_count =  len(datasets)
        # 分割获得各个list

        knowledge_list = [item["generateKnowledge"] for item in datasets]
        flag_list = [item["flag"] for item in datasets]
        topic_list = [item["topic"] for item in datasets]

        #迭代运行（注意 目前先不使用迭代，而是一边运行一边过滤）
        #save_count = 1400
        # 使用两种方法，迭代处理数据
        with open(config["output_file"], "a",encoding='utf-8') as f:
            for i in tqdm(range(last_index, generate_count, request_batch_size)):
                # conflict_prompts,conflict_ids = self.get_conflict_prompts(knowledge_list[i:i + request_batch_size])
                reasonable_prompts = self.get_reasonable_prompts(knowledge_list[i:i + request_batch_size])
                # results_conflict = api_generation(conflict_prompts)
                results_reasonable = api_generation(reasonable_prompts)
                for j in range(len(reasonable_prompts)):
                #    result_conflict = results_conflict[j]
                    result_reasonable = results_reasonable[j]
                 #   response_conflict = result_conflict.get("response")
                    response_reasonable = result_reasonable.get("response")
                    index = i + j
                    # 处理response字段，将其转换为包含input和output的字典
                    # output_lines = response.split("\n")
                    # input_value = output_lines[0].replace("input: ", "")
                    # output_value = output_lines[1].replace("output: ", "")
                    # if output_value == "" or input_value == "":
                    #    index -= 1
                    #    continue
                    try:
                    #    response_conflict = json.loads(response_conflict.replace('：', ':'))
                        response_reasonable = json.loads(response_reasonable.replace('：', ':'))
                    except:
                        continue
                    record = {
                        "id": index,
                    #    "conflict_score": response_conflict['result'],
                        "reasonable_score": response_reasonable['result'],
                    #    "conflict_explanation": response_conflict['explanation'],
                        "reasonable_explanation": response_reasonable['explanation'],
                    #    "conflict_prompt": conflict_prompts[j],
                        "reasonable_prompt": reasonable_prompts[j],
                        "topic":topic_list[index]
                    #    "relation_id":conflict_ids[j]
                    }
                    embedding = embed_text(knowledge_list[i])  # 获取嵌入向量

                    # 如果 embedding 是 NumPy 数组，直接转换为 list
                    embedding_np = embedding.astype(np.float32).tolist()
                    if(response_reasonable['result']==1):
                        save_data=[
                            embedding_np,
                            [knowledge_list[index]],
                            [topic_list[index]],
                            [flag_list[index]],
                            [response_reasonable['result']]  #Confidenct
                            ]
                        self.milvusWrapperKnowledge.insert_data(save_data)
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    self.update_checkpoint(config["checkpoint_file"], index)


    def load_checkpoint(self,file_path, default=0):
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return int(f.read().strip())
        return default

    def update_checkpoint(self,file_path, index):
        with open(file_path, 'w') as f:
            f.write(str(index))

    def load_unresolved_conflicts(self, file_path):
        unresolved_conflicts = []
        with open(file_path, 'r', encoding='utf-8') as uf:
            for line in uf:
                unresolved_conflicts.append(json.loads(line))
        return unresolved_conflicts
    def update_unresolved_conflicts(self, file_path, unresolved_conflicts):
        with open(file_path, 'a', encoding='utf-8') as uf:
            uf.write(json.dumps(unresolved_conflicts, ensure_ascii=False) + '\n')

    def get_ini(self, conflict_score_string:str):
        first_digit = re.search(r'\d', conflict_score_string.content).group(0)

        return int(first_digit)

if __name__ == "__main__":
    config = get_config()
    iterateFilterProcess = IterateFilterProcess()
    get = iterateFilterProcess.process_json_data(config["validate_data_path"])
    # get = iterateFilterProcess.process_json_data("D:/PycharmProjects/TDFilter/experiment3/prestored_data.json")
    # iterateFilterProcess.pre_install(get)
    iterateFilterProcess.process(get)