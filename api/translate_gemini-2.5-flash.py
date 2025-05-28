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

# 输出路径 - 专门为gemini-2.5-flash设置
OUTPUT_BASE_DIR = '/mnt/workspace/xintong/lyx/results/AmbiTrans_api/gemini-2.5-flash'

# ========================== 模型配置 ==========================
MODEL_NAME = 'gemini-2.5-flash-preview-04-17'

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# ========================== Gemini 2.5 Flash API调用函数 ==========================
def call_api_gemini_flash(text, image, model_name):
    """专门为Gemini 2.5 Flash设计的API调用函数，关闭thinking功能"""
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
        extra_body={
            "google": {
                "thinkingConfig": {
                    "thinkingBudget": 0
                }
            }
        }
    )
    return response.choices[0].message.content

# ========================== 数据处理函数 ==========================
def get_image_folder(filename):
    if 'mma' in filename.lower():
        return IMAGE_FOLDER_MMA
    else:
        return IMAGE_FOLDER_3AM

def process_single_file(file_path, model_name, today):
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

    for item in tqdm.tqdm(data, desc=f"Processing gemini-2.5-flash on {os.path.basename(file_path)}"):
        text = user_prompt.format(en=item["en"])
        idx = item["idx"]
        image_path = os.path.join(image_folder, item["image"])

        last_error = None

        for sleep_time in sleep_times:
            try:
                outputs = call_api_gemini_flash(text, image_path, model_name)
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

    output_filename = f"gemini-2.5-flash-{today}_{os.path.basename(file_path)}"
    output_path = os.path.join(OUTPUT_BASE_DIR, output_filename)

    Path(OUTPUT_BASE_DIR).mkdir(parents=True, exist_ok=True)

    print(f"Saving results to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

# ========================== 主函数 ==========================
def main():
    print("Running Gemini 2.5 Flash translation script with thinking disabled")
    print(f"Model: {MODEL_NAME}")
    print(f"Output directory: {OUTPUT_BASE_DIR}")

    today = datetime.date.today()

    data_files = [AMBI_NORMAL_FILE, SP_FILE, MMA_FILE]

    print(f"\n{'='*50}")
    print(f"Starting processing with Gemini 2.5 Flash")
    print(f"{'='*50}")
    
    for file_path in data_files:
        if os.path.exists(file_path):
            process_single_file(file_path, MODEL_NAME, today)
        else:
            print(f"Warning: File not found: {file_path}")
    
    print(f"Completed processing with Gemini 2.5 Flash")
    print(f"All results saved in: {OUTPUT_BASE_DIR}")

if __name__ == "__main__":
    main()