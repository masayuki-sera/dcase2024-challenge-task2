import os
import librosa
import numpy as np
import soundfile as sf

# ==== パラメータ設定 ====
SOURCE_DIR = "dev/ToyTrain/train"
TARGET_DIR = "dev/ToyTrain/train_augmented_noise"
SAMPLE_RATE = 16000
NOISE_FACTOR = 0.003

os.makedirs(TARGET_DIR, exist_ok=True)

# ==== 処理開始 ====
for filename in os.listdir(SOURCE_DIR):
    if not filename.endswith(".wav"):
        continue

    src_path = os.path.join(SOURCE_DIR, filename)
    tgt_path = os.path.join(TARGET_DIR, filename.replace(".wav", "_noise.wav"))

    # 音声読み込み
    y, sr = librosa.load(src_path, sr=SAMPLE_RATE)

    # ノイズ追加
    noise = NOISE_FACTOR * np.random.randn(len(y))
    y_noisy = y + noise
    y_noisy = np.clip(y_noisy, -1.0, 1.0)  # 値のクリップ（音割れ防止）

    # 保存
    sf.write(tgt_path, y_noisy, sr)

print("ノイズ付き音声ファイルを保存しました。")
print(f"保存先: {TARGET_DIR}")
