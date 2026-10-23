import sys
import os
import torch
import torchaudio
import numpy as np
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import load_model, load_vocoder, infer_process

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

ckpt_path = "/Users/eugene/.cache/huggingface/hub/models--Misha24-10--F5-TTS_RUSSIAN/snapshots/ea166adeae4c80ec5ee423a671e2bdb83906cf84/F5TTS_v1_Base_v2/model_last_inference.safetensors"
vocab_path = "/Users/eugene/.cache/huggingface/hub/models--Misha24-10--F5-TTS_RUSSIAN/snapshots/ea166adeae4c80ec5ee423a671e2bdb83906cf84/F5TTS_v1_Base/vocab.txt"
ref_audio = "/tmp/t_clean_finish.wav"
ref_text = "Ну вот и всё, так чисто, что и плюнуть некуда."

gen_text = "Сушим весла, то есть мопы. Включаю фен на полную!"

print("Loading model...")
model = load_model(
    model_cls=DiT,
    model_cfg=dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
    ckpt_path=ckpt_path,
    vocab_file=vocab_path,
    device=device
)

print("Loading vocoder...")
vocoder = load_vocoder(vocoder_name="vocos", device=device)

print(f"Running inference for: '{gen_text}'...")
audio, sample_rate, _ = infer_process(
    ref_audio=ref_audio,
    ref_text=ref_text,
    gen_text=gen_text,
    model_obj=model,
    vocoder=vocoder,
    device=device,
    nfe_step=32,
    cfg_strength=2.0,
    speed=1.0
)

out_wav = "/tmp/test_800.wav"
audio_tensor = torch.from_numpy(audio) if isinstance(audio, np.ndarray) else torch.tensor(audio)
if audio_tensor.ndim == 1:
    audio_tensor = audio_tensor.unsqueeze(0)
if audio_tensor.dtype != torch.float32:
    audio_tensor = audio_tensor.to(torch.float32)
torchaudio.save(out_wav, audio_tensor, sample_rate)
print(f"SUCCESS! Saved generated audio to: {out_wav} (sample_rate={sample_rate})")
