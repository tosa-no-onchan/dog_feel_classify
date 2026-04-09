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

import signal

num_classes=5
num_frames = 8
max_duration = 4.0

#CLASS_NAMES = ["alert", "hungry", "miss", "log_time_no_see", "background"] # フォルダ名と一致させる
CLASS_NAMES = ["alert", "background", "hungry", "log_time_no_see", "miss"]

# 1. Queueのサイズを制限する (重要!)
# 推論待ちが溜まりすぎるとメモリがパンクするので、最大2つまでに制限
inference_queue = queue.Queue(maxsize=2)


#1. 映像の常時キャプチャ（OpenCV + deque）
#メモリ効率と速度を両立させるため、Pythonの collections.deque を使ったリングバッファが最適です。
from collections import deque

# 4秒分を保持するための設定
FPS = 15  # Orange Piでの現実的なフレームレート
MAX_LEN = FPS * int(max_duration)  # 4秒分 = 60枚


inference_worker_f=True
integrated_capture_loop_f=True


#2. 音声の常時キャプチャ（PyAudio）
#音声も同様に、直近4秒分を常にバッファしておきます。
import pyaudio

CHUNK = 1024   # 0.064 秒 分
#CHUNK = 1600  # 100ms（0.1秒）単位
RATE = 16000

THRESHOLD = 0.05  # ここは環境に合わせて調整
# 0.5 秒のチェック
#WINDOW_SIZE = 5   # 0.1秒 × 5 = 0.5秒
WINDOW_SIZE = int(0.5 / (1.0 / float(FPS))) # 7.5 -> 7

audio_buffer = deque(maxlen=int(RATE / CHUNK * max_duration)) # 4秒分のチャンク

# ターゲットとなるマイクID
MIC_ID = 3 

# リングバッファ（最大サイズを超えると古いものから自動で消える）
video_buffer = deque(maxlen=MAX_LEN)
# 常に最新0.5秒だけを保持する「予備バッファ」
pre_v_buf = deque(maxlen=WINDOW_SIZE)
pre_a_buf = deque(maxlen=WINDOW_SIZE)

#-----
# video and audio caption
#-----
def integrated_capture_loop():
    print("integrated_capture_loop(): start")
    is_recording = False

    # Camera 初期化
    cap = cv2.VideoCapture(0)
    # fps set
    cap.set(cv2.CAP_PROP_FPS, 15)
    # 低負荷にするため解像度を落としておく
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Mic 初期化
    p = pyaudio.PyAudio()
    #stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE,
    #                input=True, frames_per_buffer=CHUNK)
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE,
                input=True, input_device_index=MIC_ID,
                frames_per_buffer=CHUNK)

    print('Watching sound comming')
    while integrated_capture_loop_f:
        # 1. Video get
        #frame = get_frame()      # 0.1秒間隔で取得
        ret, frame = cap.read()

        # 1. アスペクト比維持リサイズ
        frame_resized = my_model.resize_with_padding(frame)
        # 表示確認するなら、ここ!!
        if False:
            cv2.imshow("Result", frame_resized)
            if cv2.waitKey(1) & 0xFF == 27: # ESCキー
                print("ESC検知: 停止フラグを折ります")
                caputure_f_video = False # これで while を抜ける

        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Audio get
        #audio = get_audio()      # 0.1秒分
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        #audio_buffer.append(y)

        if not is_recording:
            # --- 【待機モード】 ---
            # 常に最新0.5秒を更新し続ける（これが「過去」になる）
            pre_v_buf.append(frame_rgb)
            pre_a_buf.append(audio)

            if trigger_detected():
                print("★トリガー！(0.5秒前から録画)")
                is_recording = True
                
                # 「過去0.5秒」を本番用バッファに流し込む
                video_buffer.clear()
                audio_buffer.clear()
                video_buffer.extend(list(pre_v_buf))
                audio_buffer.extend(list(pre_a_buf))
                pre_v_buf.clear()
                pre_a_buf.clear()
                # 監視タイマーセット
                chk_start = time.time()
        else:
            # --- 【蓄積モード】 ---
            # 残りの3.5秒分を追加していく
            video_buffer.append(frame_rgb)
            audio_buffer.append(audio)
            
            if len(video_buffer) >= MAX_LEN and len(audio_buffer) >= int(RATE / CHUNK * max_duration):
                print("●4秒蓄積完了(0.5秒過去込)!")
                v_data, a_data = capture_4sec_data()
                #print('len(video_buffer):',len(video_buffer))
                #print('len(audio_buffer):',len(audio_buffer))

                # 3. Queueに放り込む
                try:
                    # block=False にすることで、推論が詰まっている時に
                    # メインループまでフリーズするのを防ぐ
                    inference_queue.put((v_data, a_data, time.ctime()), block=False)

                except queue.Full:
                    print("警告: 推論が追いついていないため、今回の enque はスキップします")
                    # 古いデータを捨てて最新を入れるなら Queue.get_nowait() してから put

                # 4.0 * 4 =16[sec] 監視を継続する
                if ((time.time() - chk_start) > max_duration*4.0):
                    pre_v_buf.extend(list(video_buffer)[-WINDOW_SIZE:])
                    pre_a_buf.extend(list(audio_buffer)[-WINDOW_SIZE:])
                    print('Watching sound comming')
                    is_recording = False

                video_buffer.clear()
                audio_buffer.clear()

    cap.release()
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("integrated_capture_loop() : #99 terminated")

#3. ここがキモ：capture_4sec_data() の中身
#トリガーが引かれた瞬間、これら2つの「動いているバッファ」から、その瞬間のデータを 「移し替え（コピー）」 します。
def capture_4sec_data():
    #print("capture_4sec_data(): called!!")
    # 映像が60枚、音声が64000サンプル(4秒)溜まるまで送らない
    if len(video_buffer) < MAX_LEN or len(audio_buffer) < int(RATE / CHUNK * max_duration):    # 4 秒
        print("data incomplete 0")
        return None
    
    # --- 映像の切り出し (60枚から8枚を間引く) ---
    current_video = list(video_buffer) # 参照を固定
    if len(current_video) < MAX_LEN:
        print("data incomplete 1")
        return None # データ不足
    
    # 60枚の中から等間隔に8枚選ぶ (C++の indices 計算と同じ)
    indices = np.linspace(0, len(current_video)-1, num_frames).astype(int)

    #print('indices:',indices)

    frames_8 = [current_video[i] for i in indices]
    # ここで NumPy化して「移し替え」完了
    v_data = my_model.preprocess_images_numpy(frames_8)

    # --- 音声の切り出し (全チャンクを結合) ---
    current_audio = list(audio_buffer)
    y_4sec = np.concatenate(current_audio) # 結合して「移し替え」完了
    a_data = my_model.preprocess_audio_numpy(y_4sec)

    if False:
        cv2.imwrite("debug_video_snap0.jpg", cv2.cvtColor(frames_8[0], cv2.COLOR_RGB2BGR))
        cv2.imwrite("debug_video_snap1.jpg", cv2.cvtColor(frames_8[1], cv2.COLOR_RGB2BGR))
        cv2.imwrite("debug_video_snap2.jpg", cv2.cvtColor(frames_8[2], cv2.COLOR_RGB2BGR))
        cv2.imwrite("debug_video_snap3.jpg", cv2.cvtColor(frames_8[3], cv2.COLOR_RGB2BGR))
        cv2.imwrite("debug_video_snap4.jpg", cv2.cvtColor(frames_8[4], cv2.COLOR_RGB2BGR))
        cv2.imwrite("debug_video_snap5.jpg", cv2.cvtColor(frames_8[5], cv2.COLOR_RGB2BGR))
        cv2.imwrite("debug_video_snap6.jpg", cv2.cvtColor(frames_8[6], cv2.COLOR_RGB2BGR))
        cv2.imwrite("debug_video_snap7.jpg", cv2.cvtColor(frames_8[7], cv2.COLOR_RGB2BGR))
        # 音声を保存 (16bit整数に戻して保存)
        import wave
        with wave.open("debug_audio_snap.wav", "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            # y_4sec は float なので戻す
            wf.writeframes((y_4sec * 32767).astype(np.int16).tobytes())

    #  v_data = np.zeros((1, 8, 3, 224, 224), dtype=np.float32)
    #  a_data = np.zeros((1, 1024, 128), dtype=np.float32)
    return v_data, a_data

def inference_worker(model_session, class_names):
    print("start inference_worker()")
    """
    推論専用スレッド: 高速コア(4スレッド)をフルに使って3.5秒の推論を行う
    """
    global inference_worker_f
    while inference_worker_f:
        try:
            # データが届くまで待機 (タイムアウト付き)
            item = inference_queue.get(timeout=1)
            if item is None: 
                print("inference_worker() : #5 get intrrupt")
                break
            
            # データを取り出す (映像テンソル, 音声テンソル, 発生時刻)
            pixel_values, input_values, timestamp = item
            
            print(f"[@{timestamp}] 推論開始...")
            #print(f"DEBUG: v_mean={pixel_values.mean():.4f}, a_mean={input_values.mean():.4f}")            
            # --- ONNX推論実行 (3.5秒の重い処理) ---
            inputs = {'video_input': pixel_values, 'audio_input': input_values}
            if True:
                outputs = model_session.run(None, inputs)
                logits = outputs[0]
                #print('logits.shape:',logits.shape)
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

def trigger_detected():
    """
    最新0.5秒間の平均RMSがしきい値を超えたらTrueを返す
    """
    # audio_bufferには常に(1600,)のNumPy配列がappendされている前提
    if len(pre_a_buf) < WINDOW_SIZE:
        return False

    # 1. 最新の5個（0.5秒分）を取り出す
    # dequeをリスト化して末尾から5つ取得
    recent_chunks = list(pre_a_buf)[-WINDOW_SIZE:]
    
    # 2. 0.5秒間の平均音圧(RMS)を計算
    # 全データを結合して計算
    combined_samples = np.concatenate(recent_chunks)
    rms = np.sqrt(np.mean(combined_samples**2))
    
    # デバッグ表示（必要なら）
    #print(f"Current RMS (0.5s): {rms:.4f}")
    if rms > THRESHOLD:
        return True
    return False

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
    # video and audio caption
    threading.Thread(target=integrated_capture_loop, daemon=True).start()
    time.sleep(1)

    # 2. 推論スレッドは daemon=False (デフォルト) で起動して join() で待つ
    inference_thread = threading.Thread(target=inference_worker, args=(session, CLASS_NAMES))
    inference_thread.start()

    # このあと、 main_monitor_loop() コールか
    #main_monitor_loop(inference_thread)
    print("start main_monitor_loop()")
    """
    メイン監視スレッド: 低速コアでマイクとカメラを監視
    """
    end_proc_ok=False
    try:
        while integrated_capture_loop_f:
            time.sleep(0.1) # CPU負荷を抑えるためのわずかな休憩
    except KeyboardInterrupt:
        print("\nCtrl+C を検知。終了処理中...")
    finally:
        integrated_capture_loop_f=False
        # ここでカメラやマイクを確実に閉じる
        # cap.release() など
        # --- ここが重要！ ---
        # 1. スレッドに終了を通知 (Noneを投げる)
        inference_queue.put(None)
        #inference_worker_f=False
        # 2. スレッドが完全に終わるまで待機
        inference_thread.join()
        end_proc_ok=True

    if not end_proc_ok:
        #caputure_f_video=False
        #caputure_f_audio=False
        integrated_capture_loop_f=False
        inference_queue.put(None)

        #inference_worker_f=False
        # 2. スレッドが完全に終わるまで待機
        inference_thread.join()
    print("リソースを解放しました。")
    sys.exit(0)
    