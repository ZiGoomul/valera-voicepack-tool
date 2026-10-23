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
GAMER_DIR = HERE / "custom_voicepack/valera_gamer_audio"
GAMER_DIR.mkdir(parents=True, exist_ok=True)

# Copy base from backup so non-personalized phrases remain intact
for f in BACKUP_DIR.glob("*.mp3"):
    dest = GAMER_DIR / f.name
    if not dest.exists():
        shutil.copy2(f, dest)

PHRASES = {
    "10.mp3": "Сервер запущен! Валера в онлайне и готов к калибровке.",
    "20.mp3": "Валера выходит из игры. Сохранение прогресса... Всем пока!",
    "110.mp3": "Дейлик активирован! Валера по расписанию пошёл фармить пыль и лутать шерсть.",
    "130.mp3": "Квест принят! Валера идёт фармить локацию. Монстры, держитесь!",
    "151.mp3": "Спецквест от Саши принят в обработку! Награда гарантирована.",
    "153.mp3": "Кулдаун на швабры! Еду на базу смыть дебаффы, потом продолжу рейд.",
    "160.mp3": "Разведка карты начата! Валера сканирует туман войны в нашей локации.",
    "201.mp3": "Локация полностью зачищена! Саша, Тёма, забирайте безупречный лут!",
    "210.mp3": "Рейд завершен! Валера идет на базу заливать ману и регенить энергию.",
    "220.mp3": "Вся карта открыта, туман войны рассеян! Валера телепортируется на базу.",
    "240.mp3": "HP и мана на нуле! Отступаю на сейвзону к базе, пока не слили.",
    "280.mp3": "Зачем толкать персонажа? Валера уже зашел в сейвзону на станции!",
    "300.mp3": "Гейм овер... Аккумулятор сел... Нажмите F, чтобы почтить память Валеры...",
    "310.mp3": "Пошла регенерация энергии! Валера заряжается.",
    "370.mp3": "Дропнул швабру во время каста! Экипируйте насадку обратно, а то броня упала.",
    "380.mp3": "Швабра заэкипирована! Бонус к влажной уборке плюс пятьдесят.",
    "390.mp3": "Сняли швабру? Переключаюсь в билд сухой зачистки.",
    "400.mp3": "Данж пройден! Снимите швабры и помойте, пока дебафф не прилип.",
    "470.mp3": "АоЕ атака по площади! Валера маневрирует, берегите хитбоксы!",
    "520.mp3": "Я на скрытой локации! Ушел в стелс под диваном от кошачьего агро!",
    "550.mp3": "Тёма, чит-коды на кнопки отключены админами! Блокировка активна.",
    "560.mp3": "Режим бога выключен. Тёма, не спамь по кнопкам, Валере еще гриндить.",
    "570.mp3": "Интерфейс залочен! Зажмите домик, чтобы разблокировать панель управления.",
    "590.mp3": "Валера попал в капкан из проводов! Снимите стан, выручайте!",
    "600.mp3": "Кот скастовал паралич на щетку! Это крит по ходовой!",
    "640.mp3": "Лут из шерсти переполнил инвентарь! Почистите фильтр Валере, перегруз!",
    "700.mp3": "Слепота на лазерах! Лидар залагал, протрите мне оптику!",
    "720.mp3": "Валера застрял на локации Тёмы! Тут лабиринт из игрушек, спасайте тиммейта!",
    "730.mp3": "Гравитация отключилась! Валера завис в воздухе, верните на текстуры пола!",
    "750.mp3": "Непроходимая текстура на пути! Кто бросил тапки, уберите коллизию!",
    "780.mp3": "Валера зашел за пределы карты! Эвакуируйте меня из красной зоны!",
    "800.mp3": "Запускаю термосушку швабр. Убираем сырость на сто процентов!",
    "801.mp3": "Валера смывает грязь на чекпоинте. Стирка швабр пошла!",
    "802.mp3": "Нахватал дебаффов от кошачьей шерсти. Еду на станцию на очищение!",
    "803.mp3": "Сброс накопленного лута в базу! Заткните уши, сейчас будет громко!",
    "804.mp3": "Кончилось зелье маны! Залейте чистой воды в базу, мыть нечем!",
    "806.mp3": "Бак с маной пропал! Саша, верните бак с чистой водой на базу!",
    "807.mp3": "Резервуар с токсичной жижей переполнен! Слейте грязную воду, пока Саша не забанила!",
    "808.mp3": "Сперли бак для грязной воды! Верните лут на базу!",
    "810.mp3": "Слот мешка для пыли пуст! Установите мешок в станцию.",
    "815.mp3": "Глюк с грязным баком: либо фулл, либо дисконнект. Проверьте станцию!",
    "816.mp3": "Откат помпы на базе! Проверьте поддон станции, у Валеры гидроудар!",
    "817.mp3": "Швабры полностью высушены! Баффы обновлены, Валера готов к бою.",
    "818.mp3": "Проблема с чистой водой: низкий уровень или бак не определен сервером.",
    "900.mp3": "Потеря связи со станцией! Зажмите две кнопки на семь секунд для реконнекта.",
    "910.mp3": "Валера застрял в текстурах ковра! Админы, телепортируйте меня на базу!",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, choices=[1, 2, 3, 4], help="Run specific batch (1..4)")
    parser.add_argument("--assemble", action="store_true", help="Assemble final zip")
    args = parser.parse_args()

    items = sorted(PHRASES.items(), key=lambda x: int(x[0].split(".")[0]))
    total = len(items)

    if args.assemble:
        output_zip = HERE / "ready_voicepacks/valera_gamer.zip"
        donor_dir = HERE / "resources/official_voice_ru"
        build_work = HERE / "work/gamer_normalized"
        shutil.rmtree(build_work, ignore_errors=True)
        build_work.mkdir(parents=True, exist_ok=True)
        for f in donor_dir.glob("*.mp3"):
            shutil.copy2(f, build_work / f.name)
        for f in GAMER_DIR.glob("*.mp3"):
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
    temp_wav = f"/tmp/temp_valera_gamer_{args.batch}.wav"

    for local_idx, (filename, text) in enumerate(batch_items, 1):
        global_idx = start_idx + local_idx
        target_mp3 = GAMER_DIR / filename
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
