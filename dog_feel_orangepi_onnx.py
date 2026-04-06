#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import onnxruntime as ort

#import torch
#import torch.nn as nn
from transformers import ViTConfig, ViTModel, ASTConfig, ASTModel
import random  # これを追加
import os
import sys
#from transformers import get_linear_schedule_with_warmup

# In[ ]:
from transformers import ViTImageProcessor, ASTFeatureExtractor

# 1. 映像用プロセッサ (ViTの入力形式 224x224, 正規化などを担当)
video_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# 2. 音声用エキストラクター (ASTのメルスペクトログラム変換を担当)
audio_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")


num_classes=5

#num_frames = 16 
num_frames = 8
#max_duration = 3.0
max_duration = 4.0

CLASS_NAMES = ["alert", "hungry", "miss", "log_time_no_see", "background"] # フォルダ名と一致させる


# In[ ]:

import cv2
#import torch
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

#MODEL_PATH = "output-16frame3sec/best_loss_multimodal_model.pth"  # 保存したモデルのパス
#MODEL_PATH = "output-8frame3sec/best_loss_multimodal_model.pth"  # 保存したモデルのパス
MODEL_PATH = "/home/nishi/Documents/Visualstudio-torch_env/dog_feel_classify/dog_model_fixed-8_4.onnx"  # 保存したモデルのパス

# Int8 quant
#MODEL_PATH = "/home/nishi/Documents/Visualstudio-torch_env/dog_feel_classify/multimodal_model_quant.onnx" # 量子化版を指定

# ONNXセッションの初期化 (CPU専用)
# Orange PiのCPUリソースをフル活用するため、セッションオプションを設定
options = ort.SessionOptions()
options.intra_op_num_threads = 4  # Orange Piのコア数に合わせて調整
session = ort.InferenceSession(MODEL_PATH, 
                                sess_options=options, 
                                providers=['CPUExecutionProvider'])

print("OK")

#sys.exit()

# label, score = predict_video("test_video.mp4", model)
# print(f"判定結果: {label} (確信度: {score:.2f})")

# In[ ]:

def predict_video_onnx_cpu(video_path, num_frames=8, max_duration=4.0):
    # --- 前処理 (映像・音声ともにPyTorch版と同じ) ---
    # ... (中略: pixel_values と input_values を作成する工程) ...
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

    # --- 映像 ---
    pixel_values = video_processor(
        images=frames, 
        return_tensors="pt", # 再び pt に戻す（または np でも可）
        do_resize=False,
        do_rescale=True, 
        do_normalize=True
    ).pixel_values


    #print('type(pixel_values):',type(pixel_values))
    # バッチ次元追加
    pixel_values = np.expand_dims(pixel_values, axis=0)
    #print('pixel_values.shape',pixel_values.shape)
    #pixel_values = torch.from_numpy(pixel_values)
    #print('type(pixel_values):',type(pixel_values))

    #pixel_values = pixel_values.unsqueeze(0) 

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

    # --- 音声 ---
    input_values = audio_extractor(
        y, 
        sampling_rate=16000, 
        return_tensors="pt" # ライブラリ内部で torch を使うので pt に戻す
    ).input_values


    # ONNX用にnumpy形式に変換
    #onnx_video = pixel_values.cpu().numpy()
    onnx_video = pixel_values
    #onnx_audio = input_values.cpu().numpy()
    onnx_audio = input_values.numpy()
 
    # --- ONNX推論実行 ---
    #start_model = time.time()
    
    # 入力名の指定が必要
    inputs = {
        'video_input': onnx_video,
        'audio_input': onnx_audio
    }
    # outputs[0] がクラス判定のロジット
    outputs = session.run(None, inputs)
    
    #end_model = time.time()

    # ソフトマックス等の後処理は numpy で実装
    logits = outputs[0]
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    idx = np.argmax(probs)
    
    #print(f"実機(CPU) 推論時間: {(end_model - start_model)*1000:.2f} ms")

    return CLASS_NAMES[idx], probs[0][idx]


def predict_video_onnx(video_path, num_frames=8, max_duration=4.0):
    # --- 前処理 (映像・音声ともにPyTorch版と同じ) ---
    # ... (中略: pixel_values と input_values を作成する工程) ...
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


    # ONNX用にnumpy形式に変換
    onnx_video = pixel_values.cpu().numpy()
    onnx_audio = input_values.cpu().numpy()

    # --- ONNX推論実行 ---
    start_model = time.time()
    
    # 入力名の指定が必要
    inputs = {
        'video_input': onnx_video,
        'audio_input': onnx_audio
    }
    # outputs[0] がクラス判定のロジット
    outputs = session.run(None, inputs)
    
    end_model = time.time()
    
    # --- 後処理 ---
    logits = torch.from_numpy(outputs[0]) # Softmax計算のために一時的にTensorへ
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    confidence, predicted_idx = torch.max(probabilities, 1)

    print(f"ONNX Model inference time: {(end_model - start_model)*1000:.2f} ms")
    return CLASS_NAMES[predicted_idx.item()], confidence.item()


# In[ ]:


# --- 1. ファイルパスとラベルのリストを作成 ---
data_dir = "dataset_h264/miss"

flist=os.listdir(data_dir)
cnt=0

import time

# In[ ]:

video_path=data_dir+'/'+flist[cnt]
if True:
    for dir in CLASS_NAMES:
        data_dir = os.path.join("dataset_h264", dir)
        flist=os.listdir(data_dir)
        p_num = min(len(flist),100)
        print('-----')
        for i in range(p_num):
            video_path=data_dir+'/'+flist[i]
            print("video_path:",video_path)
            #cnt+=1
            result, confidence=predict_video_onnx_cpu(video_path, num_frames=num_frames, max_duration=max_duration)
            print('result:',result, 'confidence:',confidence)


if False:
    # --- ウォームアップ (最初の数回は計測対象外にする) ---
    for _ in range(5):
        cnt=0
        print('-------')
        for _ in range(6):
            video_path=data_dir+'/'+flist[cnt]
            #print("video_path:",video_path)
            cnt+=1
            result, confidence=predict_video_onnx_cpu(video_path, num_frames=num_frames, max_duration=max_duration)
            #print('result:',result, 'confidence:',confidence)


# In[ ]:




