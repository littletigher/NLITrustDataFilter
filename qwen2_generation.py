from tool.Filter.Contradiction_Filter import ContradictionFilter
from tool.Filter.Reasonable_Filter import ReasonableFilter
from tool.LangchainHelper.langchainMilvusHelper import LangchainMilvusHelper
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv
from tool.PromptHelper.qwen2_api import api_generation
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
        # FINE_TUNE_DATA_OUTPUT_FILE=../data_pool/fineturning_data_pool/fineturning_data.jsonl
        # LABEL_DATA_FILE=../data_pool/label_pool/labels.jsonl
        # REFERENCE_DATA_FILE=../data_pool/test_pool/biological_data_slices_labeled.jsonl
        # INSTRUCTION_DATA_FILE=../data_pool/test_pool/instruction_data_labeled.jsonl
        config_ = {
            "output_file": os.getenv("results_file"),
            "checkpoint_file": os.getenv("checkpoint_file"),
            "save_data_path": os.getenv("save_data_path"),
            # "instruction_data_file": os.getenv("INSTRUCTION_DATA_FILE"),
            "request_batch_size": int(os.getenv("FINE_TUNE_DATA_BATCH_SIZE")),
            "validate_data_path": os.getenv("validate_data_path"),
        }
        return config_
    except ValueError as e:
        print(f"环境变量配置错误: {e}")
        exit(1)
class QwenTdfProcess:
    def __init__(self):
        self.ContradictionFilter = ContradictionFilter()
        self.LangchainMilvusHelper = LangchainMilvusHelper()
        self.ResonableFilter = ReasonableFilter()

    # 提示词构建 校验矛盾与否
    def get_conflict_prompts(self, valided_datas) -> list:
        prompts_ = []
        for validata in valided_datas:
            matching_knowledge = self.get_confidence_data(validata)
            # if not matching_knowledge:
            #     continue
            prompt = (
                        f''' Based on the provided known information: {matching_knowledge}, determine whether the validation information: {validata} conflicts with it.
                         Provide a score from 1 to 5, where 1 indicates no conflict and 5 indicates a strong conflict, and 0 means the two are unrelated.only give a score of 0 if you're genuinely unsure how to rate 
                         Provide the score along with an explanation. your answer should be json format like this: {{"conflict_score": 2,"explanation"："there are no conflict"}}.'''
            )
            prompts_.append(prompt)
        return prompts_
    # 提示词构建，校验合理与否
    def get_reasonable_prompts(self, valided_datas) -> list:
        prompts_ = []
        for validata in valided_datas:
            matching_knowledge = self.get_confidence_data(validata)
            # if not matching_knowledge:
            #     continue
            prompt = (
                        f''' Please evaluate the following description based on your prior knowledge and determine if it is reasonable. Description: {validata} .Assign a score from 1 to 5, where 1 means "not reasonable,
                        " 5 means "very reasonable," and 0 means "unable to judge." Provide the score along with a rationale for your assessment. your answer should be json format like this: {{"reasonable_score": 5,"explanation"："it is reasonbale"}}.'''


            )
            prompts_.append(prompt)
        return prompts_
    def process_conflict(self):
        # 配置文件加载
        config = get_config()
        request_batch_size = config["request_batch_size"]
        last_index = self.load_checkpoint(config["checkpoint_file"])

        datasets = self.datset_preprocess(load_dataset(config["validate_data_path"]))
        with open(config["output_file"], "a") as f:
            for i in tqdm(range(last_index, len(datasets), request_batch_size)):
                batch_prompts = self.get_conflict_prompts(datasets[i:i + request_batch_size])
                results = api_generation(batch_prompts)
                for j in range(len(batch_prompts)):
                    result = results[j]
                    response = result.get("response")
                    index = i + j
                    # 处理response字段，将其转换为包含input和output的字典
                    # output_lines = response.split("\n")
                    # input_value = output_lines[0].replace("input: ", "")
                    # output_value = output_lines[1].replace("output: ", "")
                    # if output_value == "" or input_value == "":
                    #    index -= 1
                    #    continue
                    try:
                        response = json.loads(response.replace('：', ':'))
                    except:
                        continue
                    record = {
                        "id": index,
                        "prompt": batch_prompts[j],
                        "conflict_score": response['conflict_score'],
                        "explanation": response['explanation']
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    self.update_checkpoint(config["checkpoint_file"], index)
    def process_reasonable(self):
        # 配置文件加载
        config = get_config()
        request_batch_size = config["request_batch_size"]
        last_index = self.load_checkpoint(config["checkpoint_file"])

        datasets = self.datset_preprocess(load_dataset(config["validate_data_path"]))
        with open(config["output_file"], "a") as f:
            for i in tqdm(range(last_index, len(datasets), request_batch_size)):
                batch_prompts = self.get_reasonable_prompts(datasets[i:i + request_batch_size])
                results = api_generation(batch_prompts)
                for j in range(len(batch_prompts)):
                    result = results[j]
                    response = result.get("response")
                    index = i + j
                    # 处理response字段，将其转换为包含input和output的字典
                    # output_lines = response.split("\n")
                    # input_value = output_lines[0].replace("input: ", "")
                    # output_value = output_lines[1].replace("output: ", "")
                    # if output_value == "" or input_value == "":
                    #    index -= 1
                    #    continue
                    try:
                        response = json.loads(response.replace('：', ':'))
                    except:
                        continue
                    record = {
                        "id": index,
                        "prompt": batch_prompts[j],
                        "reasonable_score": response['reasonable_score'],
                        "explanation": response['explanation']
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    self.update_checkpoint(config["checkpoint_file"], index)



    def datset_preprocess(self, datasets):
        return datasets["train"]['generate_data']
    def load_data_from_huggingface(self, data: str):
        return load_dataset(data)
    def get_confidence_data(self, validate_data:str):
        return self.LangchainMilvusHelper.search_data(validate_data, 1)[0].metadata["summary"]
    def load_checkpoint(self,file_path, default=0):
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return int(f.read().strip())
        return default

    def update_checkpoint(self,file_path, index):
        with open(file_path, 'w') as f:
            f.write(str(index))

    def get_ini(self, conflict_score_string:str):
        first_digit = re.search(r'\d', conflict_score_string.content).group(0)

        return int(first_digit) # 输出：5

    def get_confidence_data(self, validate_data:str):
        return self.LangchainMilvusHelper.search_data(validate_data, 1)[0].metadata["summary"]

if __name__ == "__main__":
    process = QwenTdfProcess()
    process.process_conflict()