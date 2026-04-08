#!/usr/bin/env python
# coding: utf-8
#
# dog_feel_watch.py

import onnxruntime as ort
import random  # これを追加
import os
import sys
import threading
import queue
import time
import numpy as np
import dog_feel_orangepi_onnx as my_model
import cv2

num_classes=5
num_frames = 8
max_duration = 4.0

CLASS_NAMES = ["alert", "hungry", "miss", "log_time_no_see", "background"] # フォルダ名と一致させる

# 1. Queueのサイズを制限する (重要!)
# 推論待ちが溜まりすぎるとメモリがパンクするので、最大2つまでに制限
inference_queue = queue.Queue(maxsize=2)


#1. 映像の常時キャプチャ（OpenCV + deque）
#メモリ効率と速度を両立させるため、Pythonの collections.deque を使ったリングバッファが最適です。
from collections import deque

# 4秒分を保持するための設定
FPS = 15  # Orange Piでの現実的なフレームレート
MAX_LEN = FPS * int(max_duration)  # 4秒分 = 60枚

# リングバッファ（最大サイズを超えると古いものから自動で消える）
video_buffer = deque(maxlen=MAX_LEN)

caputure_f_video=True
caputure_f_audio=True

def video_capture_thread():
    cap = cv2.VideoCapture(0)
    # 低負荷にするため解像度を落としておく
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    global caputure_f_video
    while caputure_f_video:
        ret, frame = cap.read()
        if ret:
            # 監視スレッド側でリサイズまで済ませておくと後が楽
            frame_resized = cv2.resize(frame, (224, 224))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            video_buffer.append(frame_rgb) # ポインタを放り込むだけ

    cap.release()
    print("video_capture_thread() : #99 terminated")


#2. 音声の常時キャプチャ（PyAudio）
#音声も同様に、直近4秒分を常にバッファしておきます。
import pyaudio

CHUNK = 1024
RATE = 16000
audio_buffer = deque(maxlen=int(RATE / CHUNK * int(max_duration))) # 4秒分のチャンク

# ターゲットとなるマイクID
MIC_ID = 3 

def audio_capture_thread():
    p = pyaudio.PyAudio()
    #stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE,
    #                input=True, frames_per_buffer=CHUNK)
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE,
                input=True, input_device_index=MIC_ID,
                frames_per_buffer=CHUNK)

    print(f"マイク(ID:{MIC_ID}) 監視中... 終了はCtrl+C")
    global caputure_f_audio
    while caputure_f_audio:
        data = stream.read(CHUNK, exception_on_overflow=False)
        y = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        audio_buffer.append(y)

    stream.stop_stream()
    stream.close()
    p.terminate()
    print("audio_capture_thread() : #99 terminated")


#3. ここがキモ：capture_4sec_data() の中身
#トリガーが引かれた瞬間、これら2つの「動いているバッファ」から、その瞬間のデータを 「移し替え（コピー）」 します。

def capture_4sec_data():
    # --- 映像の切り出し (60枚から8枚を間引く) ---
    current_video = list(video_buffer) # 参照を固定
    if len(current_video) < MAX_LEN:
        return None # データ不足
    
    # 60枚の中から等間隔に8枚選ぶ (C++の indices 計算と同じ)
    indices = np.linspace(0, len(current_video)-1, 8).astype(int)
    frames_8 = [current_video[i] for i in indices]
    # ここで NumPy化して「移し替え」完了
    v_data = my_model.preprocess_images_numpy(frames_8)

    # --- 音声の切り出し (全チャンクを結合) ---
    current_audio = list(audio_buffer)
    y_4sec = np.concatenate(current_audio) # 結合して「移し替え」完了
    a_data = my_model.preprocess_audio_numpy(y_4sec)

    #  v_data = np.zeros((1, 8, 3, 224, 224), dtype=np.float32)
    #  a_data = np.zeros((1, 1024, 128), dtype=np.float32)
    return v_data, a_data

def inference_worker(model_session, class_names):
    print("start inference_worker()")
    """
    推論専用スレッド: 高速コア(4スレッド)をフルに使って3.5秒の推論を行う
    """
    while True:
        try:
            # データが届くまで待機 (タイムアウト付き)
            item = inference_queue.get(timeout=1)
            if item is None: 
                print("inference_worker() : #5 get intrrupt")
                break
            
            # データを取り出す (映像テンソル, 音声テンソル, 発生時刻)
            pixel_values, input_values, timestamp = item
            
            print(f"[@{timestamp}] 推論開始...")
            
            # --- ONNX推論実行 (3.5秒の重い処理) ---
            inputs = {'video_input': pixel_values, 'audio_input': input_values}
            if True:
                outputs = model_session.run(None, inputs)
                logits = outputs[0]
                
                # --- 5. 結果計算 (NumPy版 Softmax) ---
                exp_logits = np.exp(logits - np.max(logits))
                probs = exp_logits / exp_logits.sum()
                idx = np.argmax(probs)
                
                class_id = CLASS_NAMES[idx]
                score = probs[0][idx]
            
            # 後処理 (Softmaxなど)
            # ...結果判定...
            print(f"[@{timestamp}] 判定完了: class:{class_id} score:{score:.3f}")
            
            # 終わったら明示的に削除してメモリ解放を促す
            del pixel_values, input_values, item
            inference_queue.task_done()
            
        except queue.Empty:
            continue

# dummy
def trigger_detected():
  # 4秒間処理を停止
  time.sleep(2)
  return True

def main_monitor_loop(inference_thread):
    print("start main_monitor_loop()")
    """
    メイン監視スレッド: 低速コアでマイクとカメラを監視
    """
    try:
        while True:
            # 1. 音声トリガーを待機 (以前作成されたロジック)
            if trigger_detected():
                print("トリガー検知！サンプリング開始...")
                
                data = capture_4sec_data()
                if data is None:
                    print("データ準備中...")
                    continue # 次のループへ
                # 2. 4秒間のデータを取得 (ここでNumPy配列として作成)
                # 映像: (1, 8, 3, 224, 224) float32
                # 音声: (1, 1024, 128) float32
                v_data, a_data = data # ここで落ちなくなる
                
                # 3. Queueに放り込む
                try:
                    # block=False にすることで、推論が詰まっている時に
                    # メインループまでフリーズするのを防ぐ
                    inference_queue.put((v_data, a_data, time.ctime()), block=False)

                except queue.Full:
                    print("警告: 推論が追いついていないため、今回のトリガーはスキップします")
                    # 古いデータを捨てて最新を入れるなら Queue.get_nowait() してから put
            time.sleep(0.01) # CPU負荷を抑えるためのわずかな休憩

    except KeyboardInterrupt:
        print("\nCtrl+C を検知。終了処理中...")
    finally:
        global caputure_f_video
        global caputure_f_audio
        caputure_f_video=False
        caputure_f_audio=False
        # ここでカメラやマイクを確実に閉じる
        # cap.release() など
        # --- ここが重要！ ---
        # 1. スレッドに終了を通知 (Noneを投げる)
        inference_queue.put(None)
        # 2. スレッドが完全に終わるまで待機
        inference_thread.join()

        print("リソースを解放しました。")

if __name__ == '__main__':
  MODEL_PATH = "/home/nishi/Documents/Visualstudio-torch_env/dog_feel_classify/dog_model_fixed-8_4.onnx"  # 保存したモデルのパス

  # ONNXセッションの初期化 (CPU専用)
  # Orange PiのCPUリソースをフル活用するため、セッションオプションを設定
  options = ort.SessionOptions()
  options.intra_op_num_threads = 4  # Orange Piのコア数に合わせて調整
  session = ort.InferenceSession(MODEL_PATH, 
                                  sess_options=options, 
                                  providers=['CPUExecutionProvider'])
  print("OK")

  # スレッド起動
  # 1. 裏方スレッド（キャプチャ系）は daemon=True で起動
  threading.Thread(target=video_capture_thread, daemon=True).start()
  time.sleep(1)
  threading.Thread(target=audio_capture_thread, daemon=True).start()
  time.sleep(1)

  # 2. 推論スレッドは daemon=False (デフォルト) で起動して join() で待つ
  inference_thread = threading.Thread(target=inference_worker, args=(session, CLASS_NAMES))
  inference_thread.start()

  # このあと、 main_monitor_loop() コールか
  main_monitor_loop(inference_thread)
