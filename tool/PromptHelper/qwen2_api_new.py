import os
import dotenv
from torch import multiprocessing
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

# 全局模型和tokenizer
model = None
tokenizer = None

import os
import dotenv
from torch import multiprocessing
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime


class ModelHandler:
    def __init__(self, config):
        """
        初始化 ModelHandler 类，加载模型和tokenizer。
        """
        self.config = config
        self.device = config.get('device')
        self.model = None
        self.tokenizer = None
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        """
        加载模型和tokenizer，确保只加载一次。
        """
        if self.model is None or self.tokenizer is None:
            print("Loading model and tokenizer...")
            self.model = AutoModelForCausalLM.from_pretrained(self.config['model_path'], torch_dtype="auto",
                                                              trust_remote_code=True).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_path'], trust_remote_code=True)

    def make_request(self, prompt_):
        """
        根据提示生成响应。
        """
        messages = [{"role": "user", "content": prompt_}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(model_inputs.input_ids, max_new_tokens=self.config['max_new_tokens'])
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                         zip(model_inputs.input_ids, generated_ids)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        data_ = {"prompt": prompt_, "response": response, "created_at": str(datetime.now())}
        return data_

    def make_request_attentionMask(self, prompt_):
        """
        带有attention mask的生成请求。
        """
        messages = [{"role": "user", "content": prompt_}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(self.device)

        generated_ids = self.model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=self.config['max_new_tokens']
        )

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                         zip(model_inputs.input_ids, generated_ids)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        data_ = {"prompt": prompt_, "response": response, "created_at": str(datetime.now())}
        return data_

    def worker(self, prompts):
        """
        处理一组提示的worker函数。
        """
        results = []
        for prompt in prompts:
            result = self.make_request(prompt)
            results.append(result)
        return results


def get_config():
    """
    从环境变量中加载配置。
    """
    dotenv.load_dotenv()
    return {
        "model_path": os.getenv("llm_model_path"),
        "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS")),
        "device": os.getenv("DEVICE"),
        "max_workers": int(os.getenv("MAX_WORKERS")),
    }

config = get_config()
model_handler = ModelHandler(config)  # 实例化 ModelHandler
def api_generation(prompts):
    """
    主函数：并行处理多个提示。
    """
    device = config.get('device')
    num_processes = config.get('max_workers')

    chunked_prompts = [prompts[i::num_processes] for i in range(num_processes)]

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(model_handler.worker, [(chunk) for chunk in chunked_prompts])

    return [item for sublist in results for item in sublist]
