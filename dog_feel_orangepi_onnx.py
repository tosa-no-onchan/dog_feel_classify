#!/usr/bin/env python
# coding: utf-8
#
# dog_feel_orangepi_onnx.py
# In[ ]:
import onnxruntime as ort
import random  # これを追加
import os
import sys
from moviepy import VideoFileClip

#from scipy.signal import get_window
#from scipy.fftpack import dct

# In[ ]:
num_classes=5

#num_frames = 16 
num_frames = 8
#max_duration = 3.0
max_duration = 4.0

CLASS_NAMES = ["background","alert", "hungry", "miss", "log_time_no_see"] # フォルダ名と一致させる
# dog_feel_train.ipynb の学習時の
# classes = ["background","alert", "hungry", "miss", "log_time_no_see"] に一致させること。

import cv2
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


# label, score = predict_video("test_video.mp4", model)
# print(f"判定結果: {label} (確信度: {score:.2f})")


# In[ ]:

def preprocess_images_numpy(frames):
    """ViTImageProcessorの完全再現 (NumPy版)"""
    # 0-255 -> 0-1 & Normalize (mean=0.5, std=0.5)
    # 計算式: (x / 255.0 - 0.5) / 0.5 => x / 127.5 - 1.0
    images = np.array(frames).astype(np.float32) / 127.5 - 1.0
    # (8, 224, 224, 3) -> (1, 8, 3, 224, 224) ※軸入れ替え含む
    images = images.transpose(0, 3, 1, 2)
    return np.expand_dims(images, axis=0)


# In[ ]:

def preprocess_audio_numpy(y, max_length=1024):
    """ASTFeatureExtractorの簡易再現 (NumPy版)"""
    # librosaを使わずscipyでメルスペクトログラム計算（より軽量）
    import librosa # もしlibrosaがあればそのまま利用、なければscipyで実装可
    
    # 1. メルスペクトログラム (n_fft=400, hop=160, mels=128)
    S = librosa.feature.melspectrogram(y=y, sr=16000, n_fft=400, hop_length=160, 
                                       win_length=400, window='hamming', n_mels=128, center=False)
    # 2. Log変換 & 転置 & 正規化 (-4.26, 4.56)
    log_spec = np.log(S + 1e-10).T
    # 3. パディング/切り出し (1024固定)
    if log_spec.shape[0] < max_length:
        log_spec = np.pad(log_spec, ((0, max_length - log_spec.shape[0]), (0, 0)))
    else:
        log_spec = log_spec[:max_length, :]
    
    log_spec = (log_spec - (-4.2677393)) / (4.5689974 * 2)
    return np.expand_dims(log_spec.astype(np.float32), axis=0)

# In[ ]:


def predict_video_fast(video_path, num_frames=8, max_duration=4.0):
    # --- 1. 映像読み込み (OpenCV) ---
    cap = cv2.VideoCapture(video_path)
    # (中略: indices計算と8フレーム抽出。以前のresize_with_paddingを使用)
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
    
    #frames = [] # ここに (224,224,3) のRGB画像8枚を入れる
    #cap.release()

    # --- 2. 前処理 (NumPyのみ) ---
    pixel_values = preprocess_images_numpy(frames)
    #print('pixel_values.dtype:',pixel_values.dtype)
    
    # --- 3. 音声読み込み & 前処理 ---
    target_len = 16000 * int(max_duration)

    # --- 2. 音声抽出 (ここが NameError の原因でした) ---
    try:
        with VideoFileClip(video_path) as video:
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
    
    # y = (4秒分の音声データ)
    input_values = preprocess_audio_numpy(y)

    # --- 4. ONNX推論 ---
    start = time.time()
    outputs = session.run(None, {'video_input': pixel_values, 'audio_input': input_values})
    logits = outputs[0]
    #print('logits.shape:',logits.shape)

    # --- 5. 結果計算 (NumPy版 Softmax) ---
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    idx = np.argmax(probs)
    
    #print(f"Inference Time: {(time.time() - start)*1000:.2f} ms")
    return CLASS_NAMES[idx], probs[0][idx]



# In[ ]:

if __name__ == '__main__':

    #MODEL_PATH = "output-16frame3sec/best_loss_multimodal_model.pth"  # 保存したモデルのパス
    #MODEL_PATH = "output-8frame3sec/best_loss_multimodal_model.pth"  # 保存したモデルのパス
    #MODEL_PATH = "/home/nishi/Documents/Visualstudio-torch_env/dog_feel_classify/dog_model_fixed-8_4.onnx"  # 保存したモデルのパス
    MODEL_PATH = "/home/nishi/Documents/Visualstudio-torch_env/dog_feel_classify/dog_model_fixed-8_4-full-scartch.onnx"

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
                result, confidence=predict_video_fast(video_path, num_frames=num_frames, max_duration=max_duration)
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
                result, confidence=predict_video_fast(video_path, num_frames=num_frames, max_duration=max_duration)
                #print('result:',result, 'confidence:',confidence)


    # In[ ]:




