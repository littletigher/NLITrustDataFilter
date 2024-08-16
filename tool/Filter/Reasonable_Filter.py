
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv

from tool.LangchainHelper.localHuggingfaceData import get_huggingface_pipeline, HuggingFacePipelineSingleton

load_dotenv()
class ReasonableFilter:
    class ResultForm(BaseModel):
        confidence_score:str = Field(description="confidence score which should be int")
        # reason: str = Field(description="reason")
    def __init__(self):
        self.chain = self.init_chain()

    def invoke(self, validate_data):
        return self.chain.invoke({"validate_data": validate_data})
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
        chain = prompt_template | model
        return chain

    def init_local_model(self):
        pipeline_singleton = HuggingFacePipelineSingleton()
        pipeline = pipeline_singleton.get_pipeline()
        return pipeline
    def init_parser(self):
        return JsonOutputParser(pydantic_object=self.ResultForm)


    def init_model(self):
        return ChatOpenAI(model="gpt-3.5-turbo")

    def init_prompt_template(self):
        system_template = "Please use your prior knowledge to assess whether the data to be validated is reliable. Assign a confidence score from 1 to 5, with 5 being highly reliable, 1 being completely unreliable, and 0 if it is impossible to determine the reliability." \
                            ".the output should only contain score For example 2"
        users_template = '''
        Data to be validated: {validate_data}
        '''
        return ChatPromptTemplate.from_messages(
            [("system", system_template), ("user", users_template)]
        )
