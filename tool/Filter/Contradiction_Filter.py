
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
        conflict_score: str = Field(description="conflict score which should be int")
        explanation:str = Field(description="explanation")
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
        # model = self.init_local_model()
        model = self.init_model()
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
        system_template = "Based on the provided known information: {confidence_data}, determine whether the validation information: {validate_data} conflicts with it."
        users_template = '''
        Based on the provided known information: {confidence_data}, determine whether the validation information: {validate_data} conflicts with it.
                         Provide a score from 1 to 5, where 1 indicates no conflict and 5 indicates a strong conflict, and 0 means the two are unrelated. 
                         Provide the score along with an explanation. your answer should be json format like this: {{"conflict_score": 2,"explanation"："there are no conflict"}}
        '''
        return ChatPromptTemplate.from_messages(
            [("system", system_template), ("user", users_template)]
        )



