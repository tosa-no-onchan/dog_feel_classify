import os
import subprocess
from tqdm import tqdm

def convert_all_classes(base_input_dir, base_output_dir):
    # クラス名（フォルダ名）を取得
    classes = [d for d in os.listdir(base_input_dir) 
               if os.path.isdir(os.path.join(base_input_dir, d))]
    
    print(f"見つかったクラス: {classes}")

    for class_name in classes:
        input_class_dir = os.path.join(base_input_dir, class_name)
        output_class_dir = os.path.join(base_output_dir, class_name)
        
        # 出力先のクラスフォルダを作成
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)

        # mp4ファイルをリストアップ
        files = [f for f in os.listdir(input_class_dir) if f.endswith('.mp4')]
        
        print(f"\n--- クラス [{class_name}] を変換中 ({len(files)}件) ---")

        # 進捗バーを表示しながら変換
        for filename in tqdm(files, desc=class_name):
            input_path = os.path.join(input_class_dir, filename)
            output_path = os.path.join(output_class_dir, filename)
            go_on=False
            if not os.path.isfile(output_path):
                go_on=True
            else:
                # 更新日時を取得 (エポック秒)
                mtime1 = os.path.getmtime(input_path)
                mtime2 = os.path.getmtime(output_path)
                if mtime1 > mtime2:
                    go_on=True

            if go_on:
                # ffmpegコマンド
                # -c:v libx264 : H.264に変換
                # -crf 23 : 画質とファイルサイズのバランス設定
                # -preset faster : 変換スピードを優先（学習用なので速くてOK）
                # -c:a aac : 音声も標準的なAACに変換して安定させる
                cmd = [
                    'ffmpeg', '-y', '-i', input_path,
                    '-c:v', 'libx264', '-crf', '23', '-preset', 'faster',
                    '-c:a', 'aac', '-b:a', '128k',
                    output_path,
                    '-loglevel', 'error'
                ]
                
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"エラー発生 ({filename}): {e}")

# --- 実行 ---
# 元のフォルダ "dataset" を読み込み、新しい "dataset_h264" に書き出します
convert_all_classes("dataset", "dataset_h264")
print("\n✅ 全ての変換が完了しました！")

