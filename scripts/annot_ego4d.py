import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import cv2
import json
import decord  # compiled with cuda
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置参数
SEP_FRAME = 60
INPUT_DIR = Path(r"/mnt/qnap/data/datasets/ego4d/v1/clips")
IMAGES_DIR = Path(r"/mnt/qnap/data/datasets/ego4d_hand_sep60/images")
ANNOTATION_DIR = Path(r"/mnt/qnap/data/datasets/ego4d_hand_sep60/annotations")
GPU_ACCELERATED = True  # 启用GPU硬件加速

# 初始化MediaPipe
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=6)
detector = vision.HandLandmarker.create_from_options(options)

def detect_hands(frame: np.ndarray) -> list:
    """执行手部检测并返回标准化坐标"""
    h, w = frame.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    results = detector.detect(mp_image)

    detected_hands = []
    for hand in results.hand_landmarks:
        xx, yy = [], []
        for kps in hand:
            xx.append(kps.x)
            yy.append(kps.y)
        detected_hands.append({
            "bbox": {
                "x_min": min(xx),
                "x_max": max(xx),
                "y_min": min(yy),
                "y_max": max(yy),
            }
        })
    return detected_hands

def process_video(video_path: Path):
    """处理单个视频文件"""
    annotations = {}
    try:
        # 初始化硬件加速解码器
        ctx = decord.gpu(0) if GPU_ACCELERATED else decord.cpu(0)
        vr = decord.VideoReader(str(video_path), ctx=ctx)

        # 生成采样帧序列
        total_frames = len(vr)
        frame_indices = list(range(0, total_frames - 1, SEP_FRAME))
        if not frame_indices:
            return None

        # 批量读取视频帧（RGB格式）
        frames = vr.get_batch(frame_indices).asnumpy()

        # 创建输出目录
        video_dir = IMAGES_DIR / video_path.stem
        video_dir.mkdir(parents=True, exist_ok=True)

        for idx, frame_idx in enumerate(frame_indices):
            # 转换颜色空间为BGR用于保存
            frame_bgr = cv2.cvtColor(frames[idx], cv2.COLOR_RGB2BGR)

            # 执行检测
            try:
                hand_data = detect_hands(frames[idx])  # 使用RGB帧进行检测
            except Exception as e:
                print(f"Error at frame {frame_idx} of video {video_path.name}: {str(e)}")
                continue
            if len(hand_data) == 0:
                continue

            # 保存图像
            img_name = f"frame_{frame_idx:08d}.jpg"
            img_path = video_dir / img_name
            cv2.imwrite(str(img_path), frame_bgr)

            # 记录标注
            annotations[frame_idx] = {
                "timestamp": frame_idx / vr.get_avg_fps(),
                "image_path": str(img_path.relative_to(IMAGES_DIR)),
                "hands": hand_data
            }

    except Exception as e:
        print(f"Error processing {video_path.name}: {str(e)}")
        return None
    finally:
        del vr  # 显式释放视频读取器

    with open(os.path.join(str(ANNOTATION_DIR), f"{video_path.name.split('.')[0]}.json"), "w") as f:
        json.dump(annotations, f)
    # return {video_path.name: annotations}

def main():
    # 初始化输出目录
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

    # 获取视频文件列表
    video_files = [f for f in INPUT_DIR.iterdir()
                  if f.suffix.lower() in ('.mp4', '.avi', '.mov')]

    # 并行处理（线程池+GPU加速）
    with ThreadPoolExecutor(max_workers=4) as executor:
        # 提交所有任务到线程池
        futures = {executor.submit(process_video, video_file): video_file.name
                  for video_file in video_files}

        # 初始化带描述信息的进度条
        with tqdm(total=len(futures), desc="Processing videos") as pbar:
            for future in as_completed(futures):
                video_name = futures[future]
                try:
                    # 获取结果（虽然我们不使用，但可以捕获异常）
                    future.result()
                    pbar.set_postfix_str(video_name, refresh=False)
                except Exception as e:
                    pbar.write(f"Error processing {video_name}: {str(e)}")
                finally:
                    pbar.update(1)

if __name__ == "__main__":
    main()