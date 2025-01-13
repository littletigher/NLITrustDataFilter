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
from tool.PromptHelper.match_fake import get_random_original
from tool.PromptHelper.roberta_api import ConfidenceInferencer
from tool.NliHelper.nliHelper import ContradictionInferencer
from tool.DecisionTree.decisionTree import decisionTree
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
            "roberta_contradiction": os.getenv("roberta_contradiction"),
            "roberta_confidence": os.getenv("roberta_confidence"),
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
        # self.milvusWrapperKnowledge = MilvusWrapper(host='localhost', port='19530', collection_name='radiation_iterate_special_qwen')
        self.init_roberta()
    #
    def init_roberta(self):
        '''
        初始化roberta模型
        :return:
        '''
        config = get_config()
        self.roberta_confidence = ConfidenceInferencer()
        self.roberta_contradiction = ContradictionInferencer()
    def process_basic_roberta(self,datasets)->list:
        '''
        处理数据
        :param datasets:
        :return:
        '''
        config = get_config()
        last_index = self.load_checkpoint(config["checkpoint_file"])
        # 断点保存
        request_batch_size = config["request_batch_size"]
        generate_count = len(datasets)
        # generate_count = 200
        # 处理数据集
        knowledge_list = [item["generateKnowledge"] for item in datasets]
        flag_list = [item["flag"] for item in datasets]
        topic_list = [item["topic"] for item in datasets]
        # 迭代运行（注意 目前先不使用迭代，而是一边运行一边过滤）
        # save_count = 1400
        with open(config["output_file"], "a", encoding='utf-8') as f:
            for i in tqdm(range(last_index, generate_count, request_batch_size)):
                knowledge_process_list = []
                flag_process_list = []
                topic_process_list = []
                for j in range(request_batch_size):
                    index = i + j
                    if index >= generate_count:
                        break
                    knowledge = knowledge_list[index]
                    flag = flag_list[index]
                    topic = topic_list[index]
                    knowledge_process_list.append(knowledge)
                    flag_process_list.append(flag)
                    topic_process_list.append(topic)
                #  处理完毕后进行推理
                if (len(knowledge_process_list) == 0):
                    continue
                confidence_list = self.roberta_confidence.infer(knowledge_process_list)
                for k in range(len(flag_process_list)):
                    try:
                        record = {
                            "flag": flag_process_list[k],
                            "confidence_score": confidence_list[0][k],
                            "topic": topic_process_list[k],
                            "knowledge": knowledge_process_list[k],
                        }
                        # 将record记录在文件中
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')
                        if (confidence_list[0][k] == 1):
                            embedding = embed_text(knowledge_process_list[k])
                            embedding_np = embedding.astype(np.float32).tolist()
                            # record.update({"embedding":embedding})
                            save_data = [
                                embedding_np,
                                [knowledge_process_list[k]],
                                [topic_process_list[k]],
                                [flag_process_list[k]],
                                [confidence_list[0][k]],  # Confidenct
                            ]
                            self.milvusWrapperKnowledge.insert_data(save_data)
                            # 将record记录在数据库中
                        self.update_checkpoint(config["checkpoint_file"], index)
                    except:
                        print("error")
                        continue

    def decisionTree(self,confidence_flag, score_of_confidence, contradict_flag, score_of_contradict):

        return decisionTree(int(confidence_flag), score_of_confidence, contradict_flag, score_of_contradict)
    def process_iterator_roberta(self, datasets) -> list:
        '''
        使用处理数据，多轮过滤
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

        # topic_list = [item["topic"] for item in datasets]

        # 定义最大迭代次数，避免无限循环
        max_iterations = 3
        iteration = 0

        # 每次过滤时创建或追加文件
        with open(config["output_file"], "a", encoding='utf-8') as f:
            while iteration < max_iterations:
                iteration += 1
                print(f"开始第 {iteration} 次过滤...")

                for i in tqdm(range(last_index, generate_count, request_batch_size)):
                    knowledge_process_list=[]
                    flag_process_list=[]
                    matching_knowledge_list=[]
                    for j in range(request_batch_size):
                        index = i + j
                        if index >= generate_count:
                            break
                        knowledge = knowledge_list[index]
                        flag = flag_list[index]
                        matching_knowledge = self.get_confidence_data(knowledge, mod="local")
                        #   处理该index的数据
                        if not matching_knowledge:
                            unresolved_conflicts.append(datasets[index])
                            continue
                        knowledge_process_list.append(knowledge)
                        flag_process_list.append(flag)
                        matching_knowledge_list.append(matching_knowledge[0])
                    #  处理完毕后进行推理
                    if(len(knowledge_process_list)==0):
                        continue
                    confidence_list = self.roberta_confidence.infer(knowledge_process_list)
                    contradiction_list= self.roberta_contradiction.infer([item["knowledge"] for item in matching_knowledge_list],knowledge_process_list)
                    for k in range(len(flag_process_list)):
                        try:
                            # embedding_np,
                            # [knowledge_list[i]],
                            # ["none"],
                            # [flag_list[i]],
                            # [1.0],  # Confidenct
                            # [-1.0],  # Contradiction
                            # [1.0],
                            # [1.0],

                            record = {
                                "knowledge": knowledge_process_list[k],
                                "flag": flag_process_list[k],
                                "confidence_flag":confidence_list[0][k],
                                "contradiction_flag":contradiction_list[0][k],
                                "score_of_confidence":confidence_list[1][k],
                                "score_of_contradiction":contradiction_list[1][k],

                            }

                            # 将record记录在文件中
                            f.write(json.dumps(record, ensure_ascii=False) + '\n')
                            if (self.decisionTree(confidence_list[0][k], confidence_list[1][k],contradiction_list[0][k], contradiction_list[1][k])==1):
                                embedding = embed_text(knowledge_process_list[k])
                                embedding_np = embedding.astype(np.float32).tolist()
                                # record.update({"embedding":embedding})
                                save_data = [
                                    embedding_np,
                                    [knowledge_process_list[k]],
                                    [matching_knowledge_list[k]["knowledge"]],
                                    [flag_process_list[k]],
                                    [confidence_list[0][k]],
                                    [contradiction_list[0][k]],
                                    [confidence_list[1][k]],
                                    [contradiction_list[1][k]],
                                ]
                                self.milvusWrapperKnowledge.insert_data(save_data)
                                # 将record记录在数据库中
                            self.update_checkpoint(config["checkpoint_file"], index)
                        except:
                            print("error")
                            continue
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

                generate_count = len(datasets)
                print(datasets)
                # 处理数据集
                knowledge_list = [item["generateKnowledge"] for item in datasets]
                flag_list = [item["flag"] for item in datasets]

        # print("将最后剩下的无法判断的数据处理掉")
        self.process_iterator_oneTime_roberta(datasets)
    def process_iterator_oneTime_roberta(self, datasets) -> list:
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

        #迭代运行（注意 目前先不使用迭代，而是一边运行一边过滤）
        #save_count = 1400
        # 使用两种方法，迭代处理数据
        with open(config["output_file"], "a",encoding='utf-8') as f:
            for i in tqdm(range(last_index, generate_count, request_batch_size)):
                knowledge_process_list = []
                flag_process_list = []
                topic_process_list = []
                matching_knowledge_list = []
                for j in range(request_batch_size):
                    index = i + j
                    if index >= generate_count:
                        break
                    knowledge = knowledge_list[index]
                    flag = flag_list[index]
                    matching_knowledge = self.get_confidence_data(knowledge, mod="local")
                    if not matching_knowledge:
                        matching_knowledge = [{"id": -1, "knowledge": "no matching knowledge please give 0", "distance": -1}]
                    #   处理该index的数据
                    knowledge_process_list.append(knowledge)
                    flag_process_list.append(flag)
                    matching_knowledge_list.append(matching_knowledge[0])
                #  处理完毕后进行推理
                if (len(knowledge_process_list) == 0):
                    continue
                confidence_list = self.roberta_confidence.infer(knowledge_process_list)
                contradiction_list = self.roberta_contradiction.infer([item["knowledge"] for item in matching_knowledge_list],
                                                                     knowledge_process_list)
                for k in range(len(flag_process_list)):
                    try:
                        # embedding_np,
                        # [knowledge_list[i]],
                        # ["none"],
                        # [flag_list[i]],
                        # [1.0],  # Confidenct
                        # [-1.0],  # Contradiction
                        # [1.0],
                        # [1.0],

                        record = {
                            "knowledge": knowledge_process_list[k],
                            "flag": flag_process_list[k],
                            "confidence_flag": confidence_list[0][k],
                            "contradiction_flag": contradiction_list[0][k],
                            "score_of_confidence": confidence_list[1][k],
                            "score_of_contradiction": contradiction_list[1][k],

                        }

                        # 将record记录在文件中
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')
                        if (self.decisionTree(confidence_list[0][k], confidence_list[1][k], contradiction_list[0][k],
                                              contradiction_list[1][k]) == 1):
                            embedding = embed_text(knowledge_process_list[k])
                            embedding_np = embedding.astype(np.float32).tolist()
                            # record.update({"embedding":embedding})
                            save_data = [
                                embedding_np,
                                [knowledge_process_list[k]],
                                [matching_knowledge_list[k]["knowledge"]],
                                [flag_process_list[k]],
                                [confidence_list[0][k]],
                                [contradiction_list[0][k]],
                                [confidence_list[1][k]],
                                [contradiction_list[1][k]],
                            ]
                            self.milvusWrapperKnowledge.insert_data(save_data)
                            # 将record记录在数据库中
                        self.update_checkpoint(config["checkpoint_file"], index)
                    except:
                        print("error")
                        continue

    # 处理输入的数据 这里统一假设数据为json格式
    # 包含两个字段 一个是id 一个是文本
    # 他是json格式的数据
    def process_jsonl_data(self,filename):
        '''
        验证通过
        处理jsonl格式的数据
        :param filename: JSONL文件的路径
        :return: 解析后的数据或异常
        '''
        try:
            # 读取文件内容
            with open(filename, 'r', encoding='utf-8') as file:
                data = []
                for line in file:
                    data.append(json.loads(line.strip()))

                return data
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format")
        except FileNotFoundError:
            raise ValueError(f"File {filename} not found")

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
                f'''Based on the provided known information: {matching_knowledge[0]['knowledge']}, evaluate the logical consistency of the validation information: {validata}.
                        Use the known information to determine whether the validation information can be inferred or contradicted. If the known information is not closely related to the validation information, provide a lower score indicating lower consistency.
                        Provide a result as a float value between 0 and 1:
                        - A value close to 1 indicates the validation information is highly consistent with the known information.
                        - A value close to 0 indicates the validation information is highly contradictory to the known information.
                        - A value around 0.5 indicates the validation information is neither strongly consistent nor strongly contradictory, or unrelated.

                        Provide your decision and an explanation in the following JSON format:
                        {{"result": [0-1 float value], "explanation": "The validation information is consistent/contradictory/unrelated because .."}}
            '''
            )
            prompts_.append(prompt)
            ids_.append(id)
        return prompts_,ids_
    #
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
                        f'''Based on your prior knowledge in the field of biology, evaluate the following description and determine its validity on a scale from 0 to 1.
                            Description: {validata}.

                            Provide your result as follows:
                            - "confidence_flag": 1 if the information is valid and trustworthy, 0 if the information is invalid or untrustworthy.
                            - "score_of_confidence": A value between 0 and 1 representing the degree of confidence in your decision.

                        For example:
                            - If "confidence_flag" = 1, "score_of_confidence" should indicate how strongly you believe the information is valid.
                            - If "confidence_flag" = 0, "score_of_confidence" should indicate how confident you are that the information is invalid.

                            Return your result in the following JSON format,only return the confidence_flag and score_of_confidence dont need to return other information:
                        {{
                            \"confidence_flag\": 1 or 0,
                            \"score_of_confidence\": [0-1 value]
                        }}'''
            )
            prompts_.append(prompt)
        return prompts_
    def pre_install(self,datasets):
        knowledge_list = [item["generateKnowledge"] for item in datasets]
        flag_list = [item["flag"] for item in datasets]
        # topic_list = [item["topic"] for item in datasets]
        for i in range(0,len(datasets)):
            if(flag_list[i]==0):
                continue
            embedding = embed_text(knowledge_list[i])  #
            embedding_np = embedding.astype(np.float32).tolist()
            save_data = [
                embedding_np,
                [knowledge_list[i]],
                ["none"],
                [flag_list[i]],
                [1.0],  # Confidenct
                [-1.0],  # Contradiction
                [1.0],
                [1.0],
                  # Relation_id
            ]
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

        confidence_list = [item["confidence_flag"] for item in datasets]
        score_list = [item["score_of_confidence"] for item in datasets]

        # 定义最大迭代次数，避免无限循环
        max_iterations = 5
        iteration = 0

        # 每次过滤时创建或追加文件
        with open(config["output_file"], "a", encoding='utf-8') as f:
            while iteration < max_iterations:
                iteration += 1
                print(f"开始第 {iteration} 次过滤...")

                for i in tqdm(range(last_index, generate_count, request_batch_size)):
                    knowledge_process_list = []
                    score_process_list = []
                    flag_process_list = []
                    confidence_process_list = []
                    matching_knowledge_list = []
                    for j in range(request_batch_size):
                        index = i + j
                        if index >= generate_count:
                            break
                        knowledge = knowledge_list[index]
                        flag = flag_list[index]
                        # matching_knowledge = self.get_confidence_data(knowledge, mod="local")
                        matching_knowledge = get_random_original()
                        #   处理该index的数据
                        if not matching_knowledge:
                            unresolved_conflicts.append(datasets[index])
                            continue
                        knowledge_process_list.append(knowledge)
                        flag_process_list.append(flag)
                        confidence_process_list.append(confidence_list[index])
                        score_process_list.append(score_list[index])
                        matching_knowledge_list.append(matching_knowledge)
                    #  处理完毕后进行推理
                    if (len(knowledge_process_list) == 0):
                        continue

                    # 大语言模型api处理confidence的方法
                    # reasonable_prompts = self.get_reasonable_prompts(
                    # knowledge_process_list)
                    # confidence_list = api_generation(reasonable_prompts)

                    # roberta模型处理的方法
                    # confidence_list = self.roberta_confidence.infer(knowledge_process_list)
                    contradiction_list = self.roberta_contradiction.infer(
                       matching_knowledge_list, knowledge_process_list)

                    for k in range(len(flag_process_list)):
                        try:

                            # try:
                            # #    response_conflict = json.loads(response_conflict.replace('：', ':'))
                            #     confidence_result = json.loads(confidence_list[k]['response'].replace('：', ':'))
                            # except:
                            #     print("无法解析大模型返回结果：" + confidence_list[k]['response'])
                            #     continue

                            final_flag = self.decisionTree(confidence_process_list[k], score_process_list[k],
                                                  contradiction_list[0][k], contradiction_list[1][k])

                            record = {
                                "flag": flag_process_list[k],
                                "final_flag":int(final_flag),
                                "confidence_flag": int(confidence_process_list[k]),
                                "contradiction_flag": contradiction_list[0][k],
                                "score_of_confidence": score_process_list[k],
                                "score_of_contradiction": contradiction_list[1][k],
                                "knowledge": knowledge_process_list[k],
                                "matchknowledge": matching_knowledge_list[k],
                            }

                            # 将record记录在文件中
                            f.write(json.dumps(record, ensure_ascii=False) + '\n')
                            # if (final_flag == 1):
                            #     embedding = embed_text(knowledge_process_list[k])
                            #     embedding_np = embedding.astype(np.float32).tolist()
                            #     # record.update({"embedding":embedding})
                            #     save_data = [
                            #         embedding_np,
                            #         [knowledge_process_list[k]],
                            #         [matching_knowledge_list[k]["knowledge"]],
                            #         [flag_process_list[k]],
                            #         [confidence_process_list[k]],
                            #         [contradiction_list[0][k]],
                            #         [score_process_list[k]],
                            #         [contradiction_list[1][k]],
                            #     ]
                            #     self.milvusWrapperKnowledge.insert_data(save_data)
                                # 将record记录在数据库中
                            self.update_checkpoint(config["checkpoint_file"], index)
                        except Exception as e:
                            print("error:", e)  # 打印具体的错误信息
                            continue
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

                generate_count = len(datasets)
                print(f"还剩下：{generate_count}")
                # 处理数据集
                knowledge_list = [item["knowledge"] for item in datasets]
                flag_list = [item["flag"] for item in datasets]

                confidence_list = [item["confidence_flag"] for item in datasets]
                score_list = [item["score_of_confidence"] for item in datasets]

            # print("将最后剩下的无法判断的数据处理掉")
            # # self.process_iterator_oneTime(datasets)
            # # 处理方法为直接存入
            # for i in range(len(knowledge_list)):
            #     record = {
            #         "flag": flag_list[i],
            #         "final_flag": confidence_list[i],
            #         "confidence_flag": confidence_list[i],
            #         "contradiction_flag": -1,
            #         "score_of_confidence": score_list[i],
            #         "score_of_contradiction": 1,
            #         "knowledge": knowledge_list[k],
            #         "matchknowledge": "None",
            #     }
            #
            #     # 将record记录在文件中
            #     f.write(json.dumps(record, ensure_ascii=False) + '\n')


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
        generate_count = len(datasets)
        # 分割获得各个list

        knowledge_list = [item["knowledge"] for item in datasets]
        flag_list = [item["flag"] for item in datasets]
        confidence_list = [item["confidence_flag"] for item in datasets]
        score_list = [item["score_of_confidence"] for item in datasets]
        # 迭代运行（注意 目前先不使用迭代，而是一边运行一边过滤）
        # save_count = 1400
        # 使用两种方法，迭代处理数据
        with open(config["output_file"], "a", encoding='utf-8') as f:
            for i in tqdm(range(last_index, generate_count, request_batch_size)):
                knowledge_process_list = []
                flag_process_list = []
                topic_process_list = []
                matching_knowledge_list = []
                for j in range(request_batch_size):
                    index = i + j
                    if index >= generate_count:
                        break
                    knowledge = knowledge_list[index]
                    flag = flag_list[index]
                    matching_knowledge = self.get_confidence_data(knowledge, mod="local")
                    if not matching_knowledge:
                        matching_knowledge = [
                            {"id": -1, "knowledge": "no matching knowledge please give 0", "distance": -1}]
                    #   处理该index的数据
                    knowledge_process_list.append(knowledge)
                    flag_process_list.append(flag)
                    matching_knowledge_list.append(matching_knowledge[0])
                #  处理完毕后进行推理
                if (len(knowledge_process_list) == 0):
                    continue
                # 大语言模型api处理confidence的方法
                # reasonable_prompts = self.get_reasonable_prompts(knowledge_process_list)
                # confidence_list = api_generation(reasonable_prompts)

                # roberta模型处理的方法
                # confidence_list = self.roberta_confidence.infer(knowledge_process_list)
                contradiction_list = self.roberta_contradiction.infer(
                    [item["knowledge"] for item in matching_knowledge_list],
                    knowledge_process_list)
                for k in range(len(flag_process_list)):
                    try:

                        try:
                            #    response_conflict = json.loads(response_conflict.replace('：', ':'))
                            confidence_result = json.loads(confidence_list[k]['response'].replace('：', ':'))
                        except:
                            print("无法解析大模型返回结果：" + confidence_list[k]['response'])
                            continue

                        record = {
                            "knowledge": knowledge_process_list[k],
                            "flag": flag_process_list[k],
                            "confidence_flag": confidence_result['confidence_flag'],
                            "contradiction_flag": contradiction_list[0][k],
                            "score_of_confidence": confidence_result['score_of_confidence'],
                            "score_of_contradiction": contradiction_list[1][k],

                        }

                        # 将record记录在文件中
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')
                        if (self.decisionTree(confidence_result['confidence_flag'],
                                              confidence_result['score_of_confidence'],
                                              contradiction_list[0][k], contradiction_list[1][k]) == 1):
                            embedding = embed_text(knowledge_process_list[k])
                            embedding_np = embedding.astype(np.float32).tolist()
                            # record.update({"embedding":embedding})
                            save_data = [
                                embedding_np,
                                [knowledge_process_list[k]],
                                [matching_knowledge_list[k]["knowledge"]],
                                [flag_process_list[k]],
                                [confidence_result['confidence_flag']],
                                [contradiction_list[0][k]],
                                [confidence_result['score_of_confidence']],
                                [contradiction_list[1][k]],
                            ]
                            self.milvusWrapperKnowledge.insert_data(save_data)
                            # 将record记录在数据库中
                        self.update_checkpoint(config["checkpoint_file"], index)
                    except:
                        print("error")
                        continue
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
        # generate_count = 15000
        # 分割获得各个list

        knowledge_list = [item["knowledge"] for item in datasets]
        flag_list = [item["flag"] for item in datasets]
        #
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
                        print("无法解析大模型返回结果："+response_reasonable)
                        continue


                    record = {
                        "id": index,
                        "flag": flag_list[index],
                        "knowledge":knowledge_list[index],
                    #    "conflict_score": response_conflict['result'],
                        "confidence_flag": response_reasonable['confidence_flag'],
                    #    "conflict_explanation": response_conflict['explanation'],
                        "score_of_confidence": response_reasonable['score_of_confidence'],
                    #    "conflict_prompt": conflict_prompts[j],
                        "reasonable_prompt": reasonable_prompts[j],
                    #    "relation_id":conflict_ids[j]
                    }
                    embedding = embed_text(knowledge_list[i])  # 获取嵌入向量

                    # 如果 embedding 是 NumPy 数组，直接转换为 list
                    embedding_np = embedding.astype(np.float32).tolist()
                    if(response_reasonable['confidence_flag']==1):
                        save_data=[
                            embedding_np,
                            [knowledge_list[index]],
                            [flag_list[index]],
                            [response_reasonable['confidence_flag']],
                            [response_reasonable['score_of_confidence']]  #Confidenct
                            ]
                        self.milvusWrapperKnowledge.insert_data(save_data)
                    f.write(json.dumps(record, ensure_ascii=False) + ',\n')
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
    # get = iterateFilterProcess.process_jsonl_data("D:/PycharmProjects/TDFilter/DataSet/Radiation/train_dataset93.json")
    # iterateFilterProcess.pre_install(get)
    iterateFilterProcess.process_iterator(get)