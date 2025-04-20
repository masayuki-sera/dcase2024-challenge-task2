import os
import librosa
import numpy as np
import soundfile as sf
import random

# === 共通設定 ===
SOURCE_DIR = "dev/ToyTrain/train"
SAMPLE_RATE = 16000

# === 拡張パラメータ ===
NOISE_FACTOR = 0.003
NOISE_FACTOR_STRONG = 0.005  # 追加ノイズ強度
PITCH_STEPS = 2
GAIN_DB = 3.0
MIXUP_ALPHA = 0.4

# === 出力フォルダ ===
AUGMENTATIONS = {
    "noise": "dev/ToyTrain/train_augmented_noise",
    "noise_005": "dev/ToyTrain/train_augmented_noise_005",  # 追加
    "pitch": "dev/ToyTrain/train_augmented_pitch",
    "gain": "dev/ToyTrain/train_augmented_gain",
    "mixup": "dev/ToyTrain/train_augmented_mixup",
}

for path in AUGMENTATIONS.values():
    os.makedirs(path, exist_ok=True)

# === 音声ファイルリスト取得 ===
file_list = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".wav")]

# === 各種変換関数 ===
def add_noise(y):
    noise = NOISE_FACTOR * np.random.randn(len(y))
    return np.clip(y + noise, -1.0, 1.0)

def add_noise_strong(y):
    noise = NOISE_FACTOR_STRONG * np.random.randn(len(y))
    return np.clip(y + noise, -1.0, 1.0)

def pitch_shift_audio(y, sr, n_steps=PITCH_STEPS):
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)

def apply_gain(y):
    factor = 10 ** (GAIN_DB / 20)
    return np.clip(y * factor, -1.0, 1.0)

def mixup(y1, y2):
    lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
    min_len = min(len(y1), len(y2))
    return np.clip(lam * y1[:min_len] + (1 - lam) * y2[:min_len], -1.0, 1.0)

# === 拡張適用 ===
for filename in file_list:
    src_path = os.path.join(SOURCE_DIR, filename)
    y, sr = librosa.load(src_path, sr=SAMPLE_RATE)

    # 1. ノイズ（0.003）
    y_noise = add_noise(y)
    sf.write(os.path.join(AUGMENTATIONS["noise"], filename.replace(".wav", "_noise.wav")), y_noise, sr)

    # 2. ピッチシフト
    y_pitch = pitch_shift_audio(y, sr)
    sf.write(os.path.join(AUGMENTATIONS["pitch"], filename.replace(".wav", "_pitch.wav")), y_pitch, sr)

    # 3. ゲイン変化
    y_gain = apply_gain(y)
    sf.write(os.path.join(AUGMENTATIONS["gain"], filename.replace(".wav", "_gain.wav")), y_gain, sr)

    # 4. Mixup（別ファイルとランダムに）
    pair_filename = random.choice([f for f in file_list if f != filename])
    y2, _ = librosa.load(os.path.join(SOURCE_DIR, pair_filename), sr=SAMPLE_RATE)
    y_mix = mixup(y, y2)
    mixup_name = filename.replace(".wav", f"_mixup_{pair_filename.replace('.wav','')}.wav")
    sf.write(os.path.join(AUGMENTATIONS["mixup"], mixup_name), y_mix, sr)

    # 5. ノイズ（0.005）追加版
    y_noise_strong = add_noise_strong(y)
    sf.write(os.path.join(AUGMENTATIONS["noise_005"], filename.replace(".wav", "_noise005.wav")), y_noise_strong, sr)

print("すべてのデータ拡張が完了しました。")
