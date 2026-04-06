#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import torch
import torch.nn as nn
from transformers import ViTConfig, ViTModel, ASTConfig, ASTModel
import random  # これを追加
import os
from transformers import get_linear_schedule_with_warmup

# In[ ]:
from transformers import ViTImageProcessor, ASTFeatureExtractor
from OrangePiOptimizedTransformer import OrangePiOptimizedTransformer

# 1. 映像用プロセッサ (ViTの入力形式 224x224, 正規化などを担当)
video_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# 2. 音声用エキストラクター (ASTのメルスペクトログラム変換を担当)
audio_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes=5

#num_frames = 16 
num_frames = 8
#max_duration = 3.0
max_duration = 4.0

CLASS_NAMES = ["alert", "hungry", "miss", "log_time_no_see", "background"] # フォルダ名と一致させる

#MODEL_PATH = "output-16frame3sec/best_loss_multimodal_model.pth"  # 保存したモデルのパス
#MODEL_PATH = "output-8frame3sec/best_loss_multimodal_model.pth"  # 保存したモデルのパス
MODEL_PATH = "output-8frame4sec/best_loss_multimodal_model.pth"  # 保存したモデルのパス

# In[ ]:

import cv2
import torch
import numpy as np

def resize_with_padding(image, target_size=(224, 224)):
    h, w = image.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_w, new_h = int(w * scale), int(h * scale)
    # リサイズ
    resized = cv2.resize(image, (new_w, new_h))
    # 黒埋め用のキャンバス作成
    canvas = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    # 中央に配置
    offset_y = (target_size[0] - new_h) // 2
    offset_x = (target_size[1] - new_w) // 2
    canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized

    return canvas

# In[ ]:

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#MODEL_PATH = "output/best_loss_multimodal_model.pth"  # 保存したモデルのパス

# --- 使い方 ---
# model = YourModelClass(...) # 以前定義したモデルクラスをインスタンス化
model = OrangePiOptimizedTransformer(num_classes=num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
#model.half()  # <--- これを追加！全ての重みを float16 に変換します

print("OK")

# label, score = predict_video("test_video.mp4", model)
# print(f"判定結果: {label} (確信度: {score:.2f})")

# In[ ]:

def predict_video(video_path, model,num_frames=16, max_duration = 4.0):
    model.eval()
    # --- 映像処理 ---
    #cap = cv2.VideoCapture(video_path)
    #total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #indices = np.linspace(0, total_frames - 1, 16).astype(int) # 16フレーム固定

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) # 1秒あたりのフレーム数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # --- 映像も「最初の3秒」に限定する ---
    #max_duration = 3.0
    # 3秒分、または動画全体の短い方のフレーム数をターゲットにする
    end_frame = min(total_frames, int(max_duration * fps))

    # 0フレームから3秒地点（end_frame）の間で16枚抜く
    indices = np.linspace(0, end_frame - 1, num_frames).astype(int)

    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 1. アスペクト比維持リサイズ
            frame = resize_with_padding(frame)
            frames.append(frame)
        else:
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
    cap.release()

    # 学習時と同じ正規化 (do_resize=False)
    #pixel_values = video_processor(images=frames, return_tensors="pt", do_resize=False).pixel_values

    # Dataset内で pixel_values を作る際、do_resize=False を指定します
    pixel_values = video_processor(
        images=frames, 
        return_tensors="pt", 
        do_resize=False,       # これが重要！ video_processor 側へ通知!! 形は自分で整えたから、プロセッサはリサイズしなくていいよ。
        do_rescale=True, 
        do_normalize=True
    ).pixel_values

    #print('type(pixel_values):',type(pixel_values))
    # バッチ次元追加
    #pixel_values = np.expand_dims(pixel_values, axis=0)
    #print('pixel_values.shape',pixel_values.shape)
    #pixel_values = torch.from_numpy(pixel_values)
    #print('type(pixel_values):',type(pixel_values))

    pixel_values = pixel_values.unsqueeze(0) 

    target_len = 16000 * int(max_duration)

    # --- 2. 音声抽出 (ここが NameError の原因でした) ---
    try:
        with VideoFileClip(path) as video:
            duration = min(video.duration, 3.0)
            audio_clip = video.audio.subclipped(0, duration)
            y = audio_clip.to_soundarray(fps=16000)
            if len(y.shape) > 1:
                y = y.mean(axis=1)

            #target_len = 48000
            if len(y) < target_len:
                y = np.pad(y, (0, target_len - len(y)))
            else:
                y = y[:target_len]
    except:
        # 音声がない等のエラー時は無音を作成
        #y = np.zeros(48000)
        y = np.zeros(target_len)

    # --- 音声処理 ---
    #try:
    #    y, sr = librosa.load(video_path, sr=16000, duration=5.0)
    #    if len(y) < 16000 * 5:
    #        y = np.pad(y, (0, 16000 * 5 - len(y)))
    #except:
    #    y = np.zeros(16000 * 5) # 音声がない場合

    #input_values = audio_extractor(y, sampling_rate=16000, return_tensors="pt").input_values

    # ここで 'input_values' を確実に定義します
    #input_values = audio_extractor(y, sampling_rate=16000, return_tensors="pt").input_values.squeeze(0)
    input_values = audio_extractor(y, sampling_rate=16000, return_tensors="pt").input_values

    # --- 推論実行 ---
    with torch.no_grad():
        pixel_values = pixel_values.to(DEVICE)
        input_values = input_values.to(DEVICE)
        #pixel_values = pixel_values.to(DEVICE).half() 
        #input_values = input_values.to(DEVICE).half()

        # 2. 正確な計測（GPUの同期をとる）
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_model = time.time()

        outputs,video_out, audio_out = model(pixel_values, input_values)
        #outputs = model(pixel_values, input_values)
        #print('outputs:',outputs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_model = time.time()

        #outputs, video_out, audio_out = model(pixels, audios)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    print(f"Model core inference time: {(end_model - start_model)*1000:.2f} ms")

    result = CLASS_NAMES[predicted_idx.item()]
    return result, confidence.item()


# In[ ]:

# --- 1. ファイルパスとラベルのリストを作成 ---
data_dir = "dataset_h264/miss"

flist=os.listdir(data_dir)
cnt=0

import time

# In[ ]:

video_path=data_dir+'/'+flist[cnt]
cnt+=1
if False:
    print("video_path:",video_path)
    result, confidence=predict_video(video_path, model)
    print('result:',result, 'confidence:',confidence)

# --- ウォームアップ (最初の数回は計測対象外にする) ---
for _ in range(5):
    cnt=0
    print('-------')
    for _ in range(6):
        video_path=data_dir+'/'+flist[cnt]
        #print("video_path:",video_path)
        cnt+=1
        result, confidence=predict_video(video_path, model,num_frames=num_frames, max_duration = max_duration)
        #print('result:',result, 'confidence:',confidence)


# In[ ]:




