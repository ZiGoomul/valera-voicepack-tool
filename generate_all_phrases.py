import sys
import os
import time
import subprocess
import torch
import torchaudio
import numpy as np
from pathlib import Path
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import load_model, load_vocoder, infer_process

PHRASES = {
    "10.mp3": "Системы активированы! Папочка снова в строю, готов наводить суету.",
    "60.mp3": "Сижу, жду вайфай. Дайте уже интернет, одичал совсем!",
    "61.mp3": "Откройте приложение на телефоне, свяжитесь со мной, настроимся как надо.",
    "70.mp3": "Вайфай отвалился! Проверьте роутер, я так работать не могу.",
    "80.mp3": "Сервер не отвечает! Там что, электричество отключили? Попробуйте позже.",
    "90.mp3": "Ловлю волну, подключаюсь к сети...",
    "100.mp3": "О, интернет подъехал! Теперь мы на связи.",
    "151.mp3": "Погнали по особому плану! Сделаю всё красиво.",
    "152.mp3": "Уборка по вашему спецзаказу начинается!",
    "153.mp3": "Сначала помоем лапки на базе, потом уже будем наводить чистоту.",
    "154.mp3": "База сейчас занята своими делами. Потерпите немного.",
    "160.mp3": "Поехал изучать владения, нарисую вам план дома.",
    "200.mp3": "Продолжаю картографическую экспедицию!",
    "202.mp3": "Вся хата нанесена на карту! Всё зафиксировал.",
    "220.mp3": "Карта готова! Еду к своей любимой на базу.",
    "240.mp3": "Батарейка почти в ноль. Еду подкрепиться током, а потом доделаю.",
    "280.mp3": "Куда ты меня пихаешь? Я и так уже дома на базе!",
    "300.mp3": "Темнота наступает... Сил не осталось, я отключаюсь... Прощайте...",
    "301.mp3": "Дай сначала досыта зарядиться, а потом уже гоняй по комнатам!",
    "370.mp3": "Ой, моп отвалился! Прицепите обратно, а то буду лысиной тереть.",
    "380.mp3": "Мокрая тряпочка на месте. Теперь я во всеоружии!",
    "390.mp3": "Тряпочку сняли. Ладно, будем только пыль гонять.",
    "400.mp3": "Дело сделано! Снимите тряпки и промойте, чтобы не затухли.",
    "470.mp3": "Я маневрирую! Побереги ноги, отойди в сторонку!",
    "491.mp3": "Опа, новая комната! Сейчас зарисуем этот неизведанный уголок.",
    "492.mp3": "Вспомнил, где я! Продолжаю чертить карту.",
    "500.mp3": "Доехал куда просили. А теперь пора домой на зарядку.",
    "510.mp3": "Тут какие-то новые дебри. Пожалуй, лучше вернусь на базу.",
    "550.mp3": "Шаловливые ручки заблокированы! Кнопки больше не нажимаются.",
    "560.mp3": "Защита снята, можете снова меня тыкать.",
    "570.mp3": "Всё заблокировано! Зажми кнопку домика, если хочешь оживить.",
    "700.mp3": "Я ослеп! Протрите мне лазерные глазки, ничего не вижу!",
    "710.mp3": "Что-то с креплением мопов. Гляньте снизу, может намоталось чего.",
    "800.mp3": "Сушим весла, то есть мопы. Включаю фен на полную!",
    "801.mp3": "Пора помыть лапки. Начинаю стирку!",
    "802.mp3": "Я весь извазюкался! Еду на базу стирать тряпки.",
    "803.mp3": "Держитесь крепче! Сейчас меня будут потрошить. Три, два, один — погнали!",
    "804.mp3": "Сушняк напал! Налейте чистой водички в базу, а то мыть нечем.",
    "806.mp3": "А где чистая вода? Верните бачок на станцию!",
    "807.mp3": "Бак с помоями полон до краёв! Слейте эту жижу, дышать нечем.",
    "808.mp3": "Куда бак для грязной воды дели? Поставьте на базу обратно!",
    "810.mp3": "А куда мне мусор сбрасывать? Мешок на базу поставьте!",
    "815.mp3": "С грязным баком беда: то ли переполнен, то ли вообще не на месте!",
    "816.mp3": "В поддоне потоп или его нет вовсе! Проверьте станцию снизу.",
    "817.mp3": "Тряпочки сухие и пахнут свежестью. Сушка окончена!",
    "818.mp3": "С чистой водой напряг: либо мало, либо бак забыли воткнуть!",
    "900.mp3": "Мы с базой потеряли взаимопонимание. Зажмите кнопки на 7 секунд, помирите нас!",
    "910.mp3": "Я попал в пушистый плен! Спасите, снимите меня с ковра!",
}

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}", flush=True)

ckpt_path = "/Users/eugene/.cache/huggingface/hub/models--Misha24-10--F5-TTS_RUSSIAN/snapshots/ea166adeae4c80ec5ee423a671e2bdb83906cf84/F5TTS_v1_Base_v2/model_last_inference.safetensors"
vocab_path = "/Users/eugene/.cache/huggingface/hub/models--Misha24-10--F5-TTS_RUSSIAN/snapshots/ea166adeae4c80ec5ee423a671e2bdb83906cf84/F5TTS_v1_Base/vocab.txt"
ref_audio = "/tmp/t_clean_finish.wav"
ref_text = "Ну вот и всё, так чисто, что и плюнуть некуда."

out_dir = Path("/Users/eugene/Downloads/Valera/voicepack-tool-xiaomi-x20-pro-max/custom_voicepack/audio")
out_dir.mkdir(parents=True, exist_ok=True)

print("Loading F5-TTS model...", flush=True)
model = load_model(
    model_cls=DiT,
    model_cfg=dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
    ckpt_path=ckpt_path,
    vocab_file=vocab_path,
    device=device
)

print("Loading Vocos vocoder...", flush=True)
vocoder = load_vocoder(vocoder_name="vocos", device=device)

items = sorted(PHRASES.items(), key=lambda x: int(x[0].split(".")[0]))
total = len(items)

print(f"Starting batch generation of {total} phrases...", flush=True)
temp_wav = "/tmp/temp_gen.wav"

for idx, (filename, text) in enumerate(items, 1):
    target_mp3 = out_dir / filename
    if target_mp3.exists() and target_mp3.stat().st_size > 1000:
        print(f"[{idx}/{total}] Skipping {filename} (already exists)", flush=True)
        continue

    t0 = time.time()
    print(f"[{idx}/{total}] Generating {filename}: \"{text}\"...", flush=True)

    try:
        audio, sample_rate, _ = infer_process(
            ref_audio=ref_audio,
            ref_text=ref_text,
            gen_text=text,
            model_obj=model,
            vocoder=vocoder,
            device=device,
            nfe_step=24,
            cfg_strength=2.0,
            speed=1.0,
            show_info=lambda x: None,
        )

        audio_tensor = torch.from_numpy(audio) if isinstance(audio, np.ndarray) else torch.tensor(audio)
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        if audio_tensor.dtype != torch.float32:
            audio_tensor = audio_tensor.to(torch.float32)

        torchaudio.save(temp_wav, audio_tensor, sample_rate)

        # Convert to Xiaomi standard MP3 format
        subprocess.run([
            "/opt/homebrew/bin/ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-i", temp_wav,
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "libmp3lame",
            "-b:a", "32k",
            "-write_xing", "0",
            "-id3v2_version", "0",
            str(target_mp3)
        ], check=True)

        elapsed = time.time() - t0
        print(f"[{idx}/{total}] Saved {filename} ({target_mp3.stat().st_size} bytes, {elapsed:.1f}s)", flush=True)

    except Exception as exc:
        print(f"[{idx}/{total}] ERROR generating {filename}: {exc}", flush=True)

print("Batch generation completed!", flush=True)
