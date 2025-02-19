import cv2
import numpy as np
from tqdm import tqdm

def compute_hash(frame):
    """使用numpy计算帧的平均哈希"""
    # 转换为灰度图并调整大小到8x8
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    # 计算平均值并生成二进制哈希
    avg = resized.mean()
    binary_hash = resized > avg
    # 将二进制数组打包为64位整数
    hash_bytes = np.packbits(binary_hash.flatten())
    return int.from_bytes(hash_bytes.tobytes(), byteorder='big')

def clean_same_frames(video_path, output_video_path):
    """高效去除重复帧"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(3)), int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    hash_set = set()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tqdm(total=total_frames, desc="Processing", unit="frame") as pbar:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            # 计算哈希并检查重复
            hash_value = compute_hash(frame)
            if hash_value not in hash_set:
                hash_set.add(hash_value)
                out.write(frame)
            pbar.update(1)

    cap.release()
    out.release()

if __name__ == "__main__":
    input_video = "./videos/video.mp4"
    output_video = "./videos/video_no_duplicates_fast.mp4"
    clean_same_frames(input_video, output_video)
    print("视频去重完成，处理速度已优化")