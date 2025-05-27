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
CLAUDE_MODEL = 'anthropic.claude-3-7-sonnet-20250219-v1:0'

def encode_and_compress_image_to_base64(
    image_path,
    max_width=8000,
    max_height=8000,
    max_size_bytes=5 * 1024 * 1024,  # 5 MB
    quality=85
):
    """
    将图片压缩到指定大小以下并转为 base64 编码字符串。
    这里的大小判断针对『Base64 后的字节数』。
    
    :param image_path: 原始图片的文件路径
    :param max_width:  最大宽度（像素）
    :param max_height: 最大高度（像素）
    :param max_size_bytes: 最终 base64 字符串的最大字节限制
    :param quality:   JPEG 初始压缩质量（1~95之间）
    :return:          压缩后图片的 base64 编码字符串
    """
    with Image.open(image_path) as img:
        # -------- 1) 等比例缩放到不超过最大宽高 --------
        width, height = img.size
        scale_factor = min(max_width / width, max_height / height, 1.0)
        if scale_factor < 1.0:
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            img = img.resize((new_width, new_height), resample=Image.LANCZOS)
        
        # -------- 2) 循环：保存为JPEG -> Base64 -> 检查大小 --------
        while True:
            # 2.1) 以当前质量保存到内存 buffer
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=quality)
            data = buffer.getvalue()

            # 2.2) Base64 编码
            b64_data = base64.b64encode(data)
            b64_size = len(b64_data)

            print(f"Quality: {quality}, Size: {b64_size}, max_size: {max_size_bytes}")
            if b64_size <= max_size_bytes:
                # 若 base64 后的大小 <= 限制，则结束循环
                break

            # 如果还是太大，则继续降低质量；也可选择进一步缩小分辨率
            # 到了极限（quality 太低）可以再尝试缩放尺寸
            quality -= 5
            if quality < 5:
                # 防止无限循环或画质过差，可以在这里进行二次缩放处理，
                # 或者直接 break 强行退出，视需求而定。
                # 这里选择直接 break 做演示。
                break
        
        # 最终得到的 b64_data 即是符合限制 (或到达极限) 的编码数据
        base64_str = b64_data.decode('utf-8')

    return base64_str

# ========================== API调用函数 ==========================
def call_api(text, image, model_name):
    base64_image = encode_and_compress_image_to_base64(image)
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

# ========================== 数据处理函数 ==========================
def get_image_folder(filename):
    if 'mma' in filename.lower():
        return IMAGE_FOLDER_MMA
    else:
        return IMAGE_FOLDER_3AM

def process_single_file(file_path, today):
    print(f"Processing file: {file_path} with Claude model")

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

    for item in tqdm.tqdm(data, desc=f"Processing Claude on {os.path.basename(file_path)}"):
        text = user_prompt.format(en=item["en"])
        idx = item["idx"]
        image_path = os.path.join(image_folder, item["image"])

        last_error = None

        for sleep_time in sleep_times:
            try:
                outputs = call_api(text, image_path, CLAUDE_MODEL)
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

    output_filename = f"claude-{today}"
    output_path = os.path.join(OUTPUT_BASE_DIR, f"{output_filename}_{os.path.basename(file_path)}")

    Path(OUTPUT_BASE_DIR).mkdir(parents=True, exist_ok=True)

    print(f"Saving results to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

# ========================== 主函数 ==========================
def main():
    print("Running Claude translation script")

    today = datetime.date.today()

    data_files = [AMBI_NORMAL_FILE, SP_FILE, MMA_FILE]

    print(f"\n{'='*50}")
    print(f"Starting processing with Claude model")
    print(f"{'='*50}")
    
    for file_path in data_files:
        if os.path.exists(file_path):
            process_single_file(file_path, today)
        else:
            print(f"Warning: File not found: {file_path}")
    
    print(f"Completed processing with Claude model")
    print("\nAll processing completed!")

if __name__ == "__main__":
    main()