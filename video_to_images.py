from scipy.stats import entropy
import cv2
import os
from http import HTTPStatus
import dashscope
from dashscope.api_entities.dashscope_response import Role
from tqdm import tqdm
import imagehash
from PIL import Image

def extract_high_quality_images(video_path, interval, top_n, blur_threshold=100, similarity_threshold=0.8):
    image_folder = './images'
    filter_image_folder = './filter_images'
    
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    if not os.path.exists(filter_image_folder):
        os.makedirs(filter_image_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    images = []
    filenames = []

    with tqdm(total=frame_count, desc="Extracting frames", unit="frame") as pbar:
        for frame_number in range(0, frame_count, interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret or not is_image_blurry(frame, blur_threshold):
                continue
            filename = f"frame_{frame_number:04d}.jpg"
            img_path = os.path.join(image_folder, filename)
            cv2.imwrite(img_path, frame)
            images.append(frame)
            filenames.append(filename)
            pbar.update(interval)

    cap.release()

    # 计算熵值并排序
    entropy_values = []
    for img, filename in tqdm(zip(images, filenames), total=len(images), desc="Calculating entropy", unit="image"):
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        entropy_value = entropy(hist, base=2)
        entropy_values.append((filename, entropy_value))

    # 按熵值排序并选择前N个
    entropy_values.sort(key=lambda x: x[1], reverse=True)
    candidate_images = entropy_values[:top_n*2]  # 扩大候选集确保最终数量

    # 图像哈希函数（改用dHash）
    def image_hash(image):
        return imagehash.dhash(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))

    # 计算汉明距离阈值（similarity_threshold=0.8 -> 最大允许距离=12）
    hash_size = 8  # dhash默认8x8，产生64位哈希
    max_distance = int((1 - similarity_threshold) * hash_size**2)
    
    selected_images = []
    selected_hashes = []

    for filename, entropy_value in tqdm(candidate_images, desc="Removing duplicates", unit="image"):
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)
        current_hash = image_hash(img)
        
        # 检查是否与已选图片相似
        is_duplicate = False
        for existing_hash in selected_hashes:
            if (current_hash - existing_hash) <= max_distance:
                is_duplicate = True
                break
        
        if not is_duplicate:
            selected_images.append((filename, entropy_value))
            selected_hashes.append(current_hash)
            if len(selected_images) >= top_n:  # 达到目标数量停止
                break

    # 保存最终结果
    for filename, _ in tqdm(selected_images, desc="Saving images", unit="image"):
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)
        cv2.imwrite(os.path.join(filter_image_folder, filename), img)
        print(f'Saved: {filename}')
    
    return [os.path.join(filter_image_folder, filename) for filename, _ in selected_images]
def conversation_with_messages(input_text):
    messages = [{'role': Role.SYSTEM, 'content': '你现在是一个文本总结助手,将杂乱的文本转换为markdown格式,输出格式要规整,简短精练,有明确markdown语法,尽量简短'},
                {'role': Role.USER, 'content': input_text}]
    response = dashscope.Generation.call(
        'qwen1.5-0.5b-chat',
        messages=messages,
        result_format='message',
    )
    if response.status_code == HTTPStatus.OK:
        return response.output['choices'][0]['message']['content']
    else:
        error_info = f"Request id: {response.request_id}, Status code: {response.status_code}, " \
                     f"error code: {response.code}, error message: {response.message}"
        return error_info

def is_image_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm > threshold

def export_images():
    video_path = "./videos/video_no_duplicates.mp4"
    interval = 5
    top_n = 20
    
    # 添加similarity_threshold参数（0.8表示过滤相似度超过80%的图片）
    extract_high_quality_images(
        video_path, 
        interval, 
        top_n,
        blur_threshold=100,
        similarity_threshold=0.8
    )
    
if __name__ == "__main__":
    export_images()