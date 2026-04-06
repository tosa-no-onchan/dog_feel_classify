import os
import subprocess
from tqdm import tqdm

def convert_to_h264(input_dir, output_dir):
    # 出力先フォルダがなければ作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # mp4ファイルをリストアップ
    files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
    
    print(f"{len(files)} 件のファイルを変換中...")

    for filename in files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # ffmpegコマンドの組み立て
        # -y: 上書き許可
        # -c:v libx264: 映像をH.264に変換
        # -c:a copy: 音声はそのままコピー
        # -crf 23: 標準的な画質設定
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-c:v', 'libx264', '-crf', '23',
            '-c:a', 'copy', output_path,
            '-loglevel', 'error' # 変換中の余計なログを出さない
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"エラー発生 ({filename}): {e}")

# 実行例 (dataset/happy を変換して dataset_h264/happy に保存)
convert_to_h264("dataset/happy", "dataset_h264/happy")


