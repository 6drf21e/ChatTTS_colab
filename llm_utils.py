try:
    import openai
except ImportError:
    print("The 'openai' module is not installed. Please install it using 'pip install openai'.")
    exit(1)
import json
import re
import time
from tqdm import tqdm
from config import LLM_RETRIES, LLM_REQUEST_INTERVAL, LLM_RETRY_DELAY, LLM_MAX_TEXT_LENGTH, LLM_PROMPT


def send_request(client, prompt, text, model):
    text = remove_json_escape_characters(text)
    messages = [{"role": "user", "content": f"{prompt}\n\n{text}"}]
    try:
        response = client.chat.completions.create(model=model, messages=messages, max_tokens=4096)
        print(response)
        return response.choices[0].message.content
    except openai.OpenAIError as e:
        print(f"OpenAI API error: {e}")
        return None


def clean_text(text):
    import re
    if isinstance(text, str):
        # 移除 ASCII 控制字符（0-31 和 127）
        text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    return text


def extract_json(response_text):
    with open("debug.txt", "w", encoding="utf8") as f:
        f.write(response_text)
    pattern = re.compile(r'((\[[^\}]{3,})?\{s*[^\}\{]{3,}?:.*\}([^\{]+\])?)', re.M | re.S)
    match = re.search(pattern, response_text)
    if match:
        return match.group(0)
    return None


def clean_and_load_json(json_string):
    try:
        cleaned_json_string = json_string.replace("'", '"')
        cleaned_json_string = clean_text(cleaned_json_string)
        # debug 写入文本
        with open("debug.json", "w", encoding="utf8") as f:
            f.write(cleaned_json_string)
        json_obj = json.loads(cleaned_json_string)
        return json_obj
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return None


def validate_json(json_obj, required_keys):
    return isinstance(json_obj, list)
    print(json_obj)
    return True
    if json_obj and all(key in json_obj for key in required_keys):
        return True
    return False


def process_text(client, prompt, text, model, required_keys):
    parts = [text[i:i + LLM_MAX_TEXT_LENGTH] for i in range(0, len(text), LLM_MAX_TEXT_LENGTH)]
    results = []

    for part in tqdm(parts, desc="Processing text"):
        for attempt in range(LLM_RETRIES + 1):
            response = send_request(client, prompt, part, model)
            if response:
                json_string = extract_json(response)
                if json_string:
                    json_obj = clean_and_load_json(json_string)
                    if validate_json(json_obj, required_keys):
                        results.extend(json_obj)
                        break
                    else:
                        print(f"Invalid JSON structure. Retrying ({attempt + 1}/{LLM_RETRIES})...")
                else:
                    print(f"No JSON found in response. Retrying ({attempt + 1}/{LLM_RETRIES})...")
            else:
                print(f"API request failed. Retrying ({attempt + 1}/{LLM_RETRIES})...")
            time.sleep(LLM_RETRY_DELAY)
        time.sleep(LLM_REQUEST_INTERVAL)

    return results


def llm_operation(api_base, api_key, model, prompt, text, required_keys):
    client = openai.OpenAI(api_key=api_key, base_url=api_base)
    return process_text(client, prompt, text, model, required_keys)


def remove_json_escape_characters(s):
    """
    移除用户提交文本中容易被llm输出导致json校验出错的字符
    :param s:
    :return:
    """
    # 定义需要移除的字符
    escape_chars = {
        '"': '',
        '\\': '',
        '/': '',
        '\b': '',
        '\f': '',
        '\n': '',
        '\r': '',
        '\t': '',
    }
    escape_re = re.compile('|'.join(re.escape(key) for key in escape_chars.keys()))

    def replace(match):
        return escape_chars[match.group(0)]

    return escape_re.sub(replace, s)
