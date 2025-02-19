import os
from tqdm import tqdm
from dashscope.api_entities.dashscope_response import Role
import re
from paddlex import create_pipeline
import jieba.analyse
from urllib.parse import quote  
from openai import OpenAI
from configs import config
def split_text_by_tokens(text):
    sentences = re.split(r'(?<=[。！？])', text)
    chunks = [sentence.strip() for sentence in sentences if sentence.strip()]
    return chunks

def extract_keywords(text, topK=10):
    keywords = jieba.analyse.extract_tags(text, topK=topK, withWeight=False)
    return keywords

def load_texts(file_path):
    texts = []
    if not os.path.exists(file_path):
        print(f"路径不存在: {file_path}")
        return texts

    files_to_process = [file_path] if os.path.isfile(file_path) else [
        os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith('.txt')
    ]

    for file_path in files_to_process:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                contents = f.read()
            chunks = split_text_by_tokens(contents)
            for chunk in chunks:
                if chunk.strip():
                    texts.append(chunk.strip())
        except UnicodeDecodeError:
            print(f"无法用UTF-8解码文件: {file_path}")
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")
    
    return texts

def generate_image_caption_based_on_keywords(keywords):
    system_prompt = "你现在是一个图像解说助手，请根据以下关键词为一张图像生成一段解说词,紧紧围绕关键词，少说废话"
    user_prompt = " ".join(keywords)

    try:
        # 调用 OpenAI 的 chat.completions 接口
        client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY") or config['api_key'], base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        try:
            completion = client.chat.completions.create(
                model="qwen1.5-0.5b-chat",  
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.5,
                max_tokens=512
            )
            response = completion.choices[0].message.content
        except Exception as e:
            print(f"生成回答失败: {e}")
            response = "抱歉，暂时无法回答这个问题"
        # 提取生成的文本
        return response
    except Exception as e:
        print(f"Error generating caption: {e}")
        return ""

def ocr_predict(image_path):
    pipeline = create_pipeline(pipeline='ocr')
    output = pipeline.predict([image_path])
    ocr_texts = []
    for res in output:
        for text in res.get('rec_text', []):  # 使用get方法以避免KeyError
            if isinstance(text, str) and text.strip():
                ocr_texts.append(text)
    return " ".join(ocr_texts).strip()

def export_md_with_keywords_and_ocr(image_path, texts, output_path):
    images = []
    if os.path.isfile(image_path):
        images.append(image_path)
    elif os.path.isdir(image_path):
        for filename in os.listdir(image_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                full_path = os.path.join(image_path, filename).replace('\\', '/')  # 将所有反斜杠替换为正斜杠
                images.append(full_path)
    else:
        print(f"无效路径：{image_path}")
        return

    if not images:
        print("未找到可匹配的图片。")
        return
    if not texts:
        print("未找到可匹配的文本。")
        return

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    md_file_path = os.path.join(output_path, "text.md")
    with open(md_file_path, "w", encoding="utf-8") as md_file:
        for image in tqdm(images, desc="处理图片", unit="image"):
            ocr_result = ocr_predict(image)
            combined_text = " ".join(texts)  # 合并所有转录文本
            transcript_keywords = extract_keywords(combined_text)  # 提取转录文本中的关键词
            
            if ocr_result:  # 如果OCR结果非空，则同时使用OCR结果和转录文本中的关键词
                ocr_keywords = extract_keywords(ocr_result)  # 提取OCR结果中的关键词
                all_keywords = list(set(transcript_keywords + ocr_keywords))  # 合并两个关键词列表，去除重复项
                caption = generate_image_caption_based_on_keywords(all_keywords)
            else:  # 如果OCR结果为空，则直接使用转录文本中的关键词
                caption = generate_image_caption_based_on_keywords(transcript_keywords)
            
            if caption:
                # 使用quote进行URL编码并确保路径使用正斜杠
                relative_image_path = quote(os.path.relpath(image, start=output_path).replace('\\', '/'))
                md_file.write(f"![{os.path.basename(image)}]({relative_image_path})\n\n{caption}\n\n")
            else:
                relative_image_path = quote(os.path.relpath(image, start=output_path).replace('\\', '/'))
                md_file.write(f"[!{os.path.basename(image)}]({relative_image_path})\n\n未生成解说。\n\n")
    print(f"Markdown文件已导出到: {md_file_path}")
if __name__ == '__main__':
    image_path = './filter_images'  # 可以是单个图片文件或包含图片的目录
    texts_path = './transcript'  # 可以是单个文本文件或包含文本的目录
    output_path = './output'  # 输出Markdown文件的目录

    texts = load_texts(texts_path)
    export_md_with_keywords_and_ocr(image_path, texts, output_path)