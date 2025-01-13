import json
import random


def get_random_original(json_file_path = "D:\PycharmProjects\TDFilter\DataSet\Fake\Fake.json"):
    """
    从给定的 JSON 文件中随机获取一个 `original` 字段的值。

    Args:
        json_file_path (str): JSON 文件路径。

    Returns:
        str: 随机选中的 `original` 字段值。
    """
    try:
        # 读取 JSON 文件
        with open(json_file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        # 如果是单个对象，直接返回其 original 字段
        if isinstance(data, dict):
            return data.get("original", "No 'original' field found.")

        # 如果是列表，随机选择一个对象并返回其 original 字段
        elif isinstance(data, list):
            random_item = random.choice(data)
            return random_item.get("original", "No 'original' field found in the selected item.")

        else:
            return "Invalid JSON structure."

    except FileNotFoundError:
        return "JSON file not found."
    except json.JSONDecodeError:
        return "Error decoding JSON file."

if __name__ == "__main__":
    # 从 JSON 文件中随机获取一个 original 字段的值
    json_file_path = "D:\PycharmProjects\TDFilter\DataSet\Fake\Fake.json"
    random_original = get_random_original(json_file_path)
    print(random_original)