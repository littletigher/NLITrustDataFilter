import time
import concurrent.futures
import dotenv
import os

import openai
from openai import OpenAI
from datetime import datetime


def make_request(
        config_, request,
):
    response_ = None
    retry_cnt = 0
    backoff_time = 30
    client_ = OpenAI(base_url=config_['api_base_url'], api_key=config_['api_key'])
    instruction_ = request.get('instruction')
    contents_ = request.get('contents')

    messages = []
    for content_ in contents_:
        message = {"role": "system", "content": content_}
        messages.append(message)
    message = {"role": "user", "content": instruction_}
    messages.append(message)
    while retry_cnt <= 3:
        try:
            response_ = client_.chat.completions.create(
                temperature=config_['temperature'],
                top_p=config_['top_p'],
                frequency_penalty=config_['frequency_penalty'],
                presence_penalty=config_['presence_penalty'],
                n=config_['n'],
                max_tokens=config_['max_tokens'],
                messages=messages,
                model=config_['engine'],
                stop=config_['stop_sequences'],
            )
            break
        except openai.APIError as e:
            print(f"OpenAIError: {e}.")
            if "Please reduce your prompt" in str(e):
                max_tokens = int(config_['max_tokens'])
                print(f"Reducing target length to {max_tokens}, retrying...")
            else:
                print(f"Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
                backoff_time *= 1.5
            retry_cnt += 1
    choice = response_.choices[0]
    content = choice.message.content
    # 去掉响应内容中的停止符，保证jsonl转义正确
    content = content.replace("\n\n", "")
    data_ = {
        "prompt": instruction_,
        "response": content,
        "created_at": str(datetime.now()),
    }
    return data_


def get_config():
    """
    Gets configuration from environment variables.

    Returns:
    - A dictionary with configuration parameters.
    """
    try:
        dotenv.load_dotenv()
        config_ = {
            "engine": os.getenv("ENGINE"),
            "max_tokens": int(os.getenv("MAX_TOKENS")),
            "temperature": float(os.getenv("TEMPERATURE")),
            "top_p": float(os.getenv("TOP_P")),
            "frequency_penalty": float(os.getenv("FREQUENCY_PENALTY")),
            "presence_penalty": float(os.getenv("PRESENCE_PENALTY")),
            "stop_sequences": os.getenv("STOP_SEQUENCES"),
            "n": int(os.getenv("N")),
            # "best_of": int(os.getenv("BEST_OF")),
            "api_key": os.getenv("OPENAI_API_KEY"),
            "api_base_url": os.getenv("OPENAI_BASE_URL"),
        }
        return config_
    except ValueError as e:
        print(f"环境变量配置错误: {e}")


def api_generation(requests):
    config = get_config()
    # 将请求进行分组，进行批处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(make_request, config, request): idx
            for idx, request in enumerate(requests)
        }
        results = [None] * len(requests)
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
    return results

if __name__ == "__main__":
    get = api_generation([{"instruction": "Hello, how are you?", "contents": ["I am fine, thank you."]}])
    print(api_generation(get))