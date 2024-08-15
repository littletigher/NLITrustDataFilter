
from tool.Filter.Contradiction_Filter import ContradictionFilter
from tool.Filter.Reasonable_Filter import ReasonableFilter
from tool.LangchainHelper.langchainMilvusHelper import LangchainMilvusHelper
from datasets import load_dataset
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
        return self.ContradictionFilter.invoke(confidence_data, validate_data)
    def reasonableInvoke(self,validate_data:str,confidence_score:int=3):
        return self.ResonableFilter.invoke(validate_data)

    def load_data_from_huggingface(self, data: str):
        return load_dataset(data)
    def save_data_to_local(self, data:list, path: str):
        # 读取环境变量,获取路径
        path = os.getenv("save_data_path")
        with open(path, "w") as f:
            json.dump(data, f)

    def listProcess(self,validate_datas:[], conflict_boundry: int = 2,confidence_boundry:int=3,init:int=0):
        results = []
        for i in range(init,len(validate_datas)):
            validate_data = validate_datas[i]
            conflict_score = self.contradictionInvoke(validate_data)
            if(conflict_score <= conflict_boundry):
                # 该数据验证合格
                is_conflict = True
            else:
                is_conflict = False
            confidence_score = self.reasonableInvoke(validate_data)
            if(confidence_score >= confidence_boundry or confidence_score==0):
                # 该数据confidence验证 合格
                is_confidence = True
            else:
                is_confidence = False
            results.append({"data":validate_data,"conflict_score":conflict_score,"confidence_score":confidence_score,"is_conflict":is_conflict,"is_confidence":is_confidence})

        return results

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
    # op = tdfProcess.contradictionInvoke("The lawmakers in Nepal have decided to elect a highly political figure as the country's first president since it became a republic.")
    # print(op)
    validate_datas = tdfProcess.load_data_from_huggingface(data="ZhJiHo/compatible_dataset")
    tdfProcess.listProcess(validate_datas["train"],conflict_boundry=2,confidence_boundry=3)
    print(validate_datas["train"])