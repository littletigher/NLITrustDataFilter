
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from tool.LangchainHelper.localHuggingfaceData import get_huggingface_pipeline, HuggingFacePipelineSingleton
from dotenv import load_dotenv
load_dotenv()
class ContradictionFilter:
    class ResultForm(BaseModel):
        conflict_score: int = Field(description="conflict score only output sore")
    def __init__(self):
        self.chain = self.init_chain()

    def invoke(self, confidence_data, validate_data):
        return self.chain.invoke({"confidence_data": confidence_data,
                      "validate_data": validate_data})
    def filter(self, condifence_data:str, validate_data:str, conflict_score:int=3):
        op = self.invoke(condifence_data, validate_data)
        if op["conflict_score"] >= conflict_score:
            return False
        return True
    def init_chain(self):
        prompt_template = self.init_prompt_template()
        model = self.init_local_model()
        parser = self.init_parser()
        chain = prompt_template | model | parser
        return chain
    def init_parser(self):
        return JsonOutputParser(pydantic_object=self.ResultForm)

    def init_local_model(self):
        pipeline_singleton = HuggingFacePipelineSingleton()
        pipeline = pipeline_singleton.get_pipeline()
        return pipeline
    def init_model(self):
        return ChatOpenAI(model="gpt-3.5-turbo")

    def init_prompt_template(self):
        system_template = "Please use known high-confidence data to verify whether the data to be validated conflicts with the existing data. You output ONLY need to be int,If there is a conflict, then the data is unreliable. If there is no conflict, then the data is relatively reliable. Please assign a conflict score from 1 to 5, with 5 indicating a conflict, 1 indicating no conflict, and 0 if the two are unrelated."
        users_template = '''
        high-confidence data: {confidence_data}
        Data to be validated: {validate_data}
        '''
        return ChatPromptTemplate.from_messages(
            [("system", system_template), ("user", users_template)]
        )


