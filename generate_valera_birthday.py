import sys
import os
import time
import shutil
import zipfile
import subprocess
import argparse
import torch
import torchaudio
import numpy as np
from pathlib import Path
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import load_model, load_vocoder, infer_process

HERE = Path("/Users/eugene/Downloads/Valera/voicepack-tool-xiaomi-x20-pro-max")
BACKUP_DIR = HERE / "custom_voicepack/audio_backup_first_pass"
BDAY_DIR = HERE / "custom_voicepack/valera_birthday_audio"
BDAY_DIR.mkdir(parents=True, exist_ok=True)

# Copy base from backup so non-personalized phrases remain intact
for f in BACKUP_DIR.glob("*.mp3"):
    dest = BDAY_DIR / f.name
    if not dest.exists():
        shutil.copy2(f, dest)

PHRASES = {
    "10.mp3": "Внимание всем игрокам сервера! Тёма празднует юбилей, десятый левел взят! Валера на связи!",
    "20.mp3": "Праздничный стрим окончен! Валера уходит на сохранение. Тёма, с днём рождения!",
    "110.mp3": "Праздничный дейлик! Валера зачищает танцпол для именинника Тёмы и гостей!",
    "130.mp3": "Квест Юбилей Тёмы! Валера выкатывается готовить комнату к поздравлениям и торту!",
    "151.mp3": "Спецзаказ от мамы Саши для именинника! Тёма, принимай идеальную комнату!",
    "153.mp3": "Кулдаун на швабры! Еду на базу смыть дебаффы, скоро вернусь на праздник Тёмы!",
    "160.mp3": "Сканирую праздничную локацию! Нарисую карту подарков и сладостей Тёмы.",
    "201.mp3": "Локация сияет! Главный босс — торт со свечами! Тёма, с десятилетием!",
    "210.mp3": "Уборка завершена! Валера едет на базу пить электрический лимонад за здоровье Тёмы!",
    "220.mp3": "План праздничного замка Тёмы готов! Валера возвращается на базу.",
    "240.mp3": "Энергия на нуле! Еду на станцию за подзарядкой, чтобы поздравить Тёму с новыми силами.",
    "280.mp3": "Не толкайтесь в очереди за тортом! Валера уже в сейвзоне на базе!",
    "300.mp3": "Батарейка села... Тёма, задуй свечи за Валеру... С юбилеем, друг...",
    "310.mp3": "Пошла праздничная подзарядка! Валера заряжается позитивом!",
    "370.mp3": "Дропнул насадку, как деталь лего! Прицепите швабру обратно к Валере!",
    "380.mp3": "Моп заэкипирован! Праздничная влажная уборка активирована!",
    "390.mp3": "Сняли швабру? Перехожу в режим сухой охоты за конфетти!",
    "400.mp3": "Праздничный рейд закончен! Снимите и помойте швабры, пожалуйста.",
    "470.mp3": "Осторожно, едет праздничный кортеж! Тёма, кот, собака — поберегите ноги и лапы!",
    "520.mp3": "Тёма, я тут! Спрятался под диваном с твоим секретным подарком!",
    "550.mp3": "Тёма, я знаю, что тебе уже десять, но кнопки сегодня на блокировке! Магии не будет!",
    "560.mp3": "Блокировка снята! Тёма, управляй аккуратно, Валере еще праздновать!",
    "570.mp3": "Кнопки заблокированы! Зажмите домик, чтобы активировать именинный режим.",
    "590.mp3": "Валера намотал праздничную ленту или шнур! Тёма, спасай тиммейта!",
    "600.mp3": "Щетка застряла! Кот, даже в день рождения Тёмы ты устраиваешь диверсии?",
    "640.mp3": "Шерсти набралось на целого крипера! Почистите фильтр Валере, дышать нечем!",
    "700.mp3": "Лазеры заляпаны праздничным кремом! Протрите лидар Валере, ничего не вижу!",
    "720.mp3": "Валера попал в минное поле из деталек Лего! Тёма, спасай напарника из завала!",
    "730.mp3": "Гравитация отключилась! Валера парит в невесомости, поставьте на пол!",
    "750.mp3": "Препятствие на пути! Кто бросил коробку от подарков? Валере не проехать!",
    "780.mp3": "Валера заехал в запретную зону с подарками! Эвакуируйте меня отсюда!",
    "800.mp3": "Сушим швабры горячим воздухом! Чистота уровня десять из десяти!",
    "801.mp3": "Валера смывает следы праздника на чекпоинте. Стирка швабр пошла!",
    "802.mp3": "Запачкался в кошачьей шерсти и блестках. Еду на станцию на очищение!",
    "803.mp3": "Сброс мусора в базу! Внимание, сейчас будет праздничный салют!",
    "804.mp3": "Кончилось зелье маны! Залейте чистой воды в базу, а то мыть нечем!",
    "806.mp3": "А где бак с чистой водой? Саша, верните резервуар на базу!",
    "807.mp3": "Бак с отходами полон! Слейте грязную воду, пока мама Саша не заругала!",
    "808.mp3": "Бак для грязной воды исчез! Верните контейнер на станцию!",
    "810.mp3": "Мешок для мусора отсутствует! Поставьте мешок на базу Валере.",
    "815.mp3": "Глюк с грязным баком: либо полон, либо отсутствует. Валере нужна помощь!",
    "816.mp3": "На станции лужа! Проверьте базу снизу, у Валеры промокли колеса!",
    "817.mp3": "Швабры чистые и сухие! Валера готов к новым подвигам в честь Тёмы!",
    "818.mp3": "Проблема с чистой водой: либо мало налили, либо бак забыли воткнуть!",
    "900.mp3": "Потеря связи с базой! Зажмите кнопки на семь секунд, помирите нас с базой!",
    "910.mp3": "Валера застрял в ворсе ковра! Тёма, сними меня с ковра в честь праздника!",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, choices=[1, 2, 3, 4], help="Run specific batch (1..4)")
    parser.add_argument("--assemble", action="store_true", help="Assemble final zip")
    args = parser.parse_args()

    items = sorted(PHRASES.items(), key=lambda x: int(x[0].split(".")[0]))
    total = len(items)

    if args.assemble:
        output_zip = HERE / "ready_voicepacks/valera_birthday.zip"
        donor_dir = HERE / "resources/official_voice_ru"
        build_work = HERE / "work/birthday_normalized"
        shutil.rmtree(build_work, ignore_errors=True)
        build_work.mkdir(parents=True, exist_ok=True)
        for f in donor_dir.glob("*.mp3"):
            shutil.copy2(f, build_work / f.name)
        for f in BDAY_DIR.glob("*.mp3"):
            shutil.copy2(f, build_work / f.name)
        with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for f in sorted(build_work.glob("*.mp3")):
                z.write(f, f.name)
        count = len(list(build_work.glob("*.mp3")))
        print(f"PACK_ASSEMBLED: {output_zip} with {count} files, size={output_zip.stat().st_size}")
        return

    batch_map = {
        1: (0, 12),
        2: (12, 24),
        3: (24, 35),
        4: (35, 46),
    }

    start_idx, end_idx = batch_map.get(args.batch, (0, total))
    batch_items = items[start_idx:end_idx]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt_path = "/Users/eugene/.cache/huggingface/hub/models--Misha24-10--F5-TTS_RUSSIAN/snapshots/ea166adeae4c80ec5ee423a671e2bdb83906cf84/F5TTS_v1_Base_v2/model_last_inference.safetensors"
    vocab_path = "/Users/eugene/.cache/huggingface/hub/models--Misha24-10--F5-TTS_RUSSIAN/snapshots/ea166adeae4c80ec5ee423a671e2bdb83906cf84/F5TTS_v1_Base/vocab.txt"
    ref_audio = "/tmp/ref_home.wav"
    ref_text = "Где-то там моя хорошая, папочка едет домой."

    model = load_model(
        model_cls=DiT,
        model_cfg=dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
        ckpt_path=ckpt_path,
        vocab_file=vocab_path,
        device=device
    )
    vocoder = load_vocoder(vocoder_name="vocos", device=device)
    temp_wav = f"/tmp/temp_valera_bday_{args.batch}.wav"

    for local_idx, (filename, text) in enumerate(batch_items, 1):
        global_idx = start_idx + local_idx
        target_mp3 = BDAY_DIR / filename
        t0 = time.time()
        audio, sample_rate, _ = infer_process(
            ref_audio=ref_audio,
            ref_text=ref_text,
            gen_text=text,
            model_obj=model,
            vocoder=vocoder,
            device=device,
            nfe_step=24,
            cfg_strength=2.2,
            speed=0.92,
            show_info=lambda x: None
        )
        audio_tensor = torch.from_numpy(audio) if isinstance(audio, np.ndarray) else torch.tensor(audio)
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        if audio_tensor.dtype != torch.float32:
            audio_tensor = audio_tensor.to(torch.float32)
        torchaudio.save(temp_wav, audio_tensor, sample_rate)

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
        print(f"DONE [{global_idx}/{total}] {filename} in {elapsed:.1f}s: \"{text}\"", flush=True)

    print(f"BATCH_COMPLETE: batch={args.batch} count={len(batch_items)}", flush=True)

if __name__ == "__main__":
    main()
