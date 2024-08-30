import os
import dotenv
import openai
from openai import OpenAI
import threading
from datetime import datetime

dotenv.load_dotenv()
client = OpenAI()
def make_request(config_, prompt_, device=None):
    response = client.chat.completions.create(
        model=config_['model_path'],
        messages=[{"role": "user", "content": prompt_}],
        max_tokens=config_['max_new_tokens']
    )
    generated_text = response.choices[0].message.content

    data_ = {"prompt": prompt_, "response": generated_text, "created_at": str(datetime.now())}
    return data_


def worker(config, prompts, results, index):
    for prompt in prompts:
        result = make_request(config, prompt)
        results[index].append(result)


def get_config():
    dotenv.load_dotenv()
    return {
        "model_path": os.getenv("api_model_name", "gpt-3.5-turbo"),
        "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS", 50)),
        "max_workers": int(os.getenv("MAX_WORKERS", 4)),
    }


def api_generation(prompts):
    config = get_config()
    num_threads = config.get('max_workers')  # 线程数

    # 分块输入
    chunked_prompts = [prompts[i::num_threads] for i in range(num_threads)]
    results = [[] for _ in range(num_threads)]

    threads = []
    for i in range(num_threads):
        thread = threading.Thread(target=worker, args=(config, chunked_prompts[i], results, i))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    # 合并所有结果
    return [item for sublist in results for item in sublist]


if __name__ == "__main__":
    prompts = ["Hello, how are you?", "What is the capital of France?"]
    responses = api_generation(prompts)
    for response in responses:
        print(response)
