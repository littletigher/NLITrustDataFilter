
from tool.Filter.Contradiction_Filter import ContradictionFilter
from tool.Filter.Reasonable_Filter import ReasonableFilter
from tool.LangchainHelper.langchainMilvusHelper import LangchainMilvusHelper
from datasets import load_dataset
from tqdm import tqdm
import json
import os
ds = load_dataset("ZhJiHo/compatible_dataset")
class TdfProcess:
    def __init__(self):
        self.ContradictionFilter = ContradictionFilter()
        self.LangchainMilvusHelper = LangchainMilvusHelper()
        self.ResonableFilter = ReasonableFilter()
    def contradictionInvoke(self,validate_data:str, conflict_score:int=3):
        confidence_data = self.LangchainMilvusHelper.search_data(validate_data, 1)
        return self.ContradictionFilter.invoke(confidence_data[0].metadata["summary"], validate_data)
    def reasonableInvoke(self,validate_data:str,confidence_score:int=3):
        return self.ResonableFilter.invoke(validate_data)

    def load_data_from_huggingface(self, data: str):
        return load_dataset(data)

    def listProcess(self, validate_datas, conflict_boundry: int = 2, confidence_boundry: int = 3, init: int = 0, results_file='results.jsonl', checkpoint_file='checkpoint.txt'):

        # Load the last processed index from the checkpoint file
        last_index = self.load_checkpoint(checkpoint_file, init)

        with open(results_file, 'a') as file:
            for i in tqdm(range(last_index, len(validate_datas)), initial=last_index, total=len(validate_datas),
                          desc="Processing"):
                validate_data = validate_datas[i]["generate_data"]
                conflict_score = self.contradictionInvoke(validate_data)
                try:
                    is_conflict = conflict_score <= conflict_boundry
                except:
                    print(f"error:can't understand {is_conflict}")
                    is_conflict = None

                confidence_score = self.reasonableInvoke(validate_data)
                try:
                    is_confidence = confidence_score >= confidence_boundry or confidence_score == 0
                except:
                    print(f"error:can't understand {confidence_score}")
                    is_confidence = None

                result = {
                    "data": validate_data,
                    "conflict_score": conflict_score,
                    "confidence_score": confidence_score,
                    "is_conflict": is_conflict,
                    "is_confidence": is_confidence
                }

                # Write the result to the JSON Lines file
                file.write(json.dumps(result) + ',\n')

                # Update the checkpoint file after each iteration
                self.update_checkpoint(checkpoint_file, i + 1)

    def load_checkpoint(self,file_path, default=0):
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return int(f.read().strip())
        return default

    def update_checkpoint(self,file_path, index):
        with open(file_path, 'w') as f:
            f.write(str(index))

    # def contradictionListInvoke(self, validate_data_list: list, conflict_score: int = 3):
    #     results = []
    #
    #     for validate_data in validate_data_list:
    #         confidence_data = self.LangchainMilvusHelper.search_data(validate_data, 1)
    #         result = self.ContradictionFilter.invoke(confidence_data, validate_data)
    #         results.append(result)
    #
    #     return results

    # 批量化处理函数

if __name__ == "__main__":
    tdfProcess = TdfProcess()
    validate_datas = tdfProcess.load_data_from_huggingface(data=os.getenv("validate_data_path"))
    tdfProcess.listProcess(validate_datas["train"],conflict_boundry=2,confidence_boundry=3,results_file='results.jsonl',checkpoint_file='checkpoint.txt')
    print(validate_datas["train"])