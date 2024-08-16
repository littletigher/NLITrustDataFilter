from langchain_core.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from dotenv import load_dotenv
import os
load_dotenv()

def get_huggingface_pipeline():

    model_id = os.getenv("llm_model_path")
    tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id,device_map="cuda:0",trust_remote_code=True)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1024)
    model = HuggingFacePipeline(pipeline=pipe)
    return model

import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain import HuggingFacePipeline

class HuggingFacePipelineSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HuggingFacePipelineSingleton, cls).__new__(cls)
            model_id = os.getenv("llm_model_path")
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True)
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1024)
            cls._instance.model = HuggingFacePipeline(pipeline=pipe)
        return cls._instance

    def get_pipeline(self):
        return self.model


if __name__ == "__main__":
    model = get_huggingface_pipeline()
    template = """Question: {question}

    Answer: Let's think step by step."""
    prompt = PromptTemplate.from_template(template)

    chain = prompt | model

    question = "What is electroencephalography?"

    print(chain.invoke({"question": question}))
