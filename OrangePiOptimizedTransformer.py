import torch
import torch.nn as nn
from transformers import ViTConfig, ViTModel, ASTConfig, ASTModel

class OrangePiOptimizedTransformer(nn.Module):
    def __init__(self, num_classes=3):
        super(OrangePiOptimizedTransformer, self).__init__()
        
        # 1. 映像: ViT-Base (768次元)
        config_vit = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k")
        self.image_model = ViTModel(config_vit)
        
        # 2. 音声: AST (768次元)
        config_ast = ASTConfig.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        config_ast.num_hidden_layers = 4
        self.audio_model = ASTModel(config_ast)
        
        # ★ここを必ず 768 にしてください
        #embed_dim = 768 
        # 2. 自動的に次元数を取得 (Baseなら768, Tinyなら192が自動で入る)
        embed_dim = self.image_model.config.hidden_size 

        # 3. 結合層にはこの embed_dim を使う
        self.fusion = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=8, 
            batch_first=True,
            #activation="relu" # GELUよりRKNNと相性が良い add by nishi 2026.4.5 for rknn
        )
        self.classifier = nn.Linear(embed_dim * 2, num_classes)

        # ★ここを追加：音声用の強めのDropout  audio だけ、dropout を試す。 by nishi 2026.4.4
        #self.audio_dropout = nn.Dropout(0.5)
        self.audio_dropout = nn.Dropout(0.7)
        
        # --- __init__ の中に追加 ---　add by nishi 2026.4.4
        # オリジナルの classifier と同じクラス数で、入力が 768 のものを用意
        self.monitor_classifier = nn.Linear(768, num_classes) 
    
    # こちらは、5 次元 [Batch, 16, 3, 224, 224] です。
    # --- forward の修正 ---
    def forward(self, pixel_values, input_values):
        # 入力の形状: [Batch, num_frames, 3, 224, 224]
        batch_size, num_frames, channels, height, width = pixel_values.shape
        
        # 映像処理
        # バッチとフレームを混ぜて一気にViTへ
        # 映像 [Batch*Frames, 3, 224, 224] -> [Batch*Frames, 768]
        pixel_values = pixel_values.view(-1, channels, height, width) 
        img_features = self.image_model(pixel_values).last_hidden_state[:, 0, :] 

        # 元のバッチサイズとフレーム数に戻して平均をとる
        # これにより num_frames がいくつであっても [Batch, 768] に集約される
        # フレーム平均 [Batch, 768]
        video_feats = img_features.view(batch_size, num_frames, -1).mean(dim=1)
    
        # 音声 [Batch, 768]
        audio_feats = self.audio_model(input_values).last_hidden_state[:, 0, :]

        # ★ここを追加：音声特徴量に Dropout をかける audio 側だけ、dropout add by nishi 2026.4.4
        audio_feats = self.audio_dropout(audio_feats)
        
        # --- 個別の判定（モニタリング用） ---
        # self.classifier ではなく、768次元用の monitor_classifier を使う
        video_logits = self.monitor_classifier(video_feats)
        audio_logits = self.monitor_classifier(audio_feats)
    
        # --- 統合判定（オリジナル通り） ---
        # 結合 (768 と 768 なので重なるようになります)
        #combined = torch.stack([img_out, aud_out], dim=1) # 以前の img_out = video_feats です
        combined = torch.stack([video_feats, audio_feats], dim=1) # 以前の img_out = video_feats です
        fused = self.fusion(combined)
        out = fused.view(fused.size(0), -1)
        
        combined_logits = self.classifier(out) # 1536次元用
    
        # 3つの判定結果を返す
        return combined_logits, video_logits, audio_logits


