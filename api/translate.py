import openai
from openai import OpenAI
import tqdm
import json
from pathlib import Path
from PIL import Image
from io import BytesIO
import os
import sys
import base64
import time
import argparse
import datetime

# ========================== 路径配置 ==========================
# API配置文件路径
with open('/mnt/workspace/xintong/api_key.txt', 'r') as f:
    lines = f.readlines()
    
API_KEY = lines[0].strip()
BASE_URL = lines[1].strip()
openai.api_key = API_KEY
openai.base_url = BASE_URL

# 数据文件路径
DATA_DIR = '/mnt/workspace/xintong/lyx/AmbiTrans_api/data/'
AMBI_NORMAL_FILE = os.path.join(DATA_DIR, 'ambi_normal_test.json')
SP_FILE = os.path.join(DATA_DIR, 'sp_test.json')
MMA_FILE = os.path.join(DATA_DIR, 'mma_test.json')

# 图片文件夹路径
IMAGE_FOLDER_3AM = '/mnt/workspace/xintong/ambi_plus/3am_images/'
IMAGE_FOLDER_MMA = '/mnt/workspace/xintong/pjh/dataset/MMA/'

# 输出路径
OUTPUT_BASE_DIR = '/mnt/workspace/xintong/lyx/results/AmbiTrans_api'

# ========================== 模型配置 ==========================
MODELS = {
    'gpt-4o': 'gpt-4o-2024-11-20',
    'o1': 'o1-2024-12-17',
    'qvq': 'qvq-max',
    'qwen': 'qwen-vl-max',
    'gemini-2.0-flash': 'gemini-2.0-flash-001',
    'claude-3-7-sonnet': 'anthropic.claude-3-7-sonnet-20250219-v1:0',
    'gemini-2.5-flash': 'gemini-2.5-flash-preview-04-17',
    'gemini-2.5-pro': 'gemini-2.5-pro-preview-05-06'
}

STANDARD_MODELS = [
    'gpt-4o-2024-11-20',
    'o1-2024-12-17',
    'qwen-vl-max',
    'gemini-2.0-flash-001',
    'anthropic.claude-3-7-sonnet-20250219-v1:0',
    'gemini-2.5-flash-preview-04-17',
    'gemini-2.5-pro-preview-05-06'
]

STREAM_MODELS = ['qvq-max']

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# ========================== API调用函数 ==========================
def call_api_standard(text, image, model_name):
    base64_image = encode_image(image)
    response = openai.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                    {"type": "text", "text": text},
                ],
            }
        ],
    )
    return response.choices[0].message.content

def call_api_stream(text, image, model_name):
    reasoning_content = ""
    answer_content = ""
    is_answering = False
    base64_image = encode_image(image)

    completion = openai.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    },
                    {"type": "text", "text": text},
                ],
            },
        ],
        stream=True,
    )

    for chunk in completion:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
            reasoning_content += delta.reasoning_content
        else:
            if delta.content and not is_answering:
                is_answering = True
            if delta.content:
                answer_content += delta.content

    return {"reasoning": reasoning_content, "answer": answer_content}

def call_api(text, image, model_name):
    if model_name in STREAM_MODELS:
        return call_api_stream(text, image, model_name)
    elif model_name in STANDARD_MODELS:
        return call_api_standard(text, image, model_name)
    else:
        raise ValueError(f"Unknown model: {model_name}. Please add it to STANDARD_MODELS or STREAM_MODELS.")

# ========================== 数据处理函数 ==========================
def get_image_folder(filename):
    if 'mma' in filename.lower():
        return IMAGE_FOLDER_MMA
    else:
        return IMAGE_FOLDER_3AM

def process_single_file(file_path, model_key, model_name, today):
    print(f"Processing file: {file_path} with model: {model_name}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    image_folder = get_image_folder(os.path.basename(file_path))

    user_prompt = """You are a multimodal translation assistant. Your task is to translate the following English sentence into accurate Chinese by fully leveraging both the text and the accompanying image.

Use the visual information to resolve any ambiguities or vague expressions in the sentence. Your translation must reflect the most precise meaning based on the image.

Only output the final Chinese translation. Do not include any explanation.

Now translate:  
{en}"""

    sleep_times = [5, 10, 20, 40, 60]

    result = []

    for item in tqdm.tqdm(data, desc=f"Processing {model_key} on {os.path.basename(file_path)}"):
        text = user_prompt.format(en=item["en"])
        idx = item["idx"]
        image_path = os.path.join(image_folder, item["image"])

        last_error = None

        for sleep_time in sleep_times:
            try:
                outputs = call_api(text, image_path, model_name)
                break
            except Exception as e:
                last_error = e
                print(f"Error on {idx}: {e}. Retry after sleeping {sleep_time} sec...")
                if "Error code: 400" in str(e) or "Error code: 429" in str(e):
                    time.sleep(sleep_time)
                else:
                    item["error"] = str(e)
                    outputs = ""
                    break
        else:
            print(f"Skipping {idx}")
            outputs = ""
            if last_error:
                item["error"] = str(last_error)

        item["result"] = outputs
        result.append(item.copy())

    output_filename = f"{model_key}-{today}"
    output_path = os.path.join(OUTPUT_BASE_DIR, f"{output_filename}_{os.path.basename(file_path)}")

    Path(OUTPUT_BASE_DIR).mkdir(parents=True, exist_ok=True)

    print(f"Saving results to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

# ========================== 主函数 ==========================
def main():
    parser = argparse.ArgumentParser(description="Multi-model translation script")
    parser.add_argument(
        '--model',
        type=str,
        nargs='+',  
        choices=[
            'gpt-4o', 'o1', 'qvq', 'qwen',
            'gemini-2.0-flash', 'claude-3-7-sonnet',
            'gemini-2.5-flash', 'gemini-2.5-pro',
            'all'
        ],
        default=['all'],
        help="Specify which model(s) to run: gpt-4o, o1, qvq, qwen, gemini-2.0-flash, claude-3-7-sonnet, gemini-2.5-flash, gemini-2.5-pro, or all"
    )

    args = parser.parse_args()

    if 'all' in args.model:
        model_names = list(MODELS.keys())
    else:
        model_names = args.model

    print(f"Running models: {model_names}")

    today = datetime.date.today()

    data_files = [AMBI_NORMAL_FILE, SP_FILE, MMA_FILE]

    for model_key in model_names:
        model_name = MODELS[model_key]
        print(f"\n{'='*50}")
        print(f"Starting processing with model: {model_key} ({model_name})")
        print(f"{'='*50}")
        
        for file_path in data_files:
            if os.path.exists(file_path):
                process_single_file(file_path, model_key, model_name, today)
            else:
                print(f"Warning: File not found: {file_path}")
        
        print(f"Completed processing with model: {model_key}")

    print("\nAll processing completed!")

if __name__ == "__main__":
    main()
