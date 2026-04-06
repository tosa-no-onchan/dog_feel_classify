import torch
import torch.nn as nn

import numpy as np
import librosa

import OrangePiOptimizedTransformer as mymodel

# インポートをシンプルにします
from torch.nn.attention import sdpa_kernel, SDPBackend


def export_to_onnx_for_rknn(model, num_frames = 8, save_path="dog_model_fixed.onnx"):
    model.eval()
    model.to("cpu")

    dummy_video = torch.randn(1, num_frames, 3, 224, 224)
    dummy_audio = torch.randn(1, 1024, 128)

    if False:
        torch.onnx.export(
            model,
            #(pixel_values, pixel_mask), # 入力タプル
            (dummy_video, dummy_audio),
            save_path,
            export_params=True,        # 重みをファイルに書き込む
            #opset_version=14,          # DETRの演算をサポートするバージョン
            opset_version=17,          # DETRの演算をサポートするバージョン
            do_constant_folding=True,  # 定数畳み込みでグラフを最適化
            input_names=['video', 'audio'],
            output_names=['final_output', 'video_features', 'audio_features']
            # dynamic_axes はあえて指定せず、サイズを 480x480 に固定します（RKNN向け）
        )

    if True:
        torch.onnx.export(
            model, 
            (dummy_video, dummy_audio), 
            save_path,
            export_params=True,
            opset_version=14, # Transformers系は14以上が安定します
            #opset_version=17, # Transformers系は14以上が安定します
            do_constant_folding=True,
            input_names=['video_input', 'audio_input'],
            output_names=['output', 'video_feat', 'audio_feat'], # OrangePiOptimizedTransformerの戻り値に合わせる
            #dynamic_axes={'video_input': {0: 'batch_size'}, 'audio_input': {0: 'batch_size'}}
        )


    print(f"✅ RKNN変換用ONNX（3出力版）を保存しました: {save_path}")

# --- 設定 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes=5

#num_frames = 16 
num_frames = 8
#max_duration = 3.0
max_duration = 4.0

CLASS_NAMES = ["alert", "hungry", "miss", "log_time_no_see", "background"] # フォルダ名と一致させる

#MODEL_PATH = "output-16frame3sec/best_loss_multimodal_model.pth"  # 保存したモデルのパス
#MODEL_PATH = "output-8frame3sec/best_loss_multimodal_model.pth"  # 保存したモデルのパス
MODEL_PATH = "output-8frame4sec/best_loss_multimodal_model.pth"  # 保存したモデルのパス


#save_path = "dog_model_fixed-8_3.onnx"
save_path = "dog_model_fixed-8_4.onnx"

# --- 使い方 ---
# model = YourModelClass(...) # 以前定義したモデルクラスをインスタンス化
model = mymodel.OrangePiOptimizedTransformer(num_classes=num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)

export_to_onnx_for_rknn(model, num_frames = num_frames,save_path=save_path)


# 8bit 化を追加

from onnxruntime.quantization import quantize_dynamic, QuantType
import onnx


# 元のモデルパス
#model_fp32 = 'multimodal_model.onnx'
model_fp32= save_path

# 量子化後のモデルパス
model_quant = 'multimodal_model_quant.onnx'

# 1. モデルの型情報を補完するために一旦ロードして保存し直す（shape_inference対策）
onnx_model = onnx.load(model_fp32)
onnx.save(onnx_model, model_fp32)



# 動的量子化の実行
if False:
    quantize_dynamic(
        model_input=model_fp32,
        model_output=model_quant,
        per_channel=False,
        reduce_range=False,
        weight_type=QuantType.QUInt8  # 重みを8bit整数に圧縮
    )

if False:
    # 2. 動的量子化の実行（エラー対策のオプションを追加）
    quantize_dynamic(
        model_input=model_fp32,
        model_output=model_quant,
        per_channel=False,
        reduce_range=False,
        weight_type=QuantType.QUInt8,
        # ↓ これが重要：認識できない型をFLOAT32として扱うよう指定
        extra_options={'DefaultTensorType': onnx.TensorProto.FLOAT}
    )

quantize_dynamic(
    #model_input='multimodal_model.onnx',
    model_input=model_fp32,
    #model_output='multimodal_model_quant.onnx',
    model_output=model_quant,
    per_channel=False,
    reduce_range=False,
    # MatMul（行列演算）のみを対象にすることでエラーを回避
    nodes_to_quantize=None, # デフォルト
    weight_type=QuantType.QUInt8,
    extra_options={'DefaultTensorType': onnx.TensorProto.FLOAT}
)


print(f"量子化モデルを保存しました: {model_quant}")