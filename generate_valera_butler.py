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
BUTLER_DIR = HERE / "custom_voicepack/valera_butler_audio"
BUTLER_DIR.mkdir(parents=True, exist_ok=True)

# Copy base from backup so non-personalized phrases remain intact
for f in BACKUP_DIR.glob("*.mp3"):
    dest = BUTLER_DIR / f.name
    if not dest.exists():
        shutil.copy2(f, dest)

PHRASES = {
    "10.mp3": "Валера к вашим услугам. Системы поместья приведены в полную готовность.",
    "20.mp3": "С вашего позволения, Валера удаляется на покой. Спокойной ночи, господа.",
    "110.mp3": "Время неумолимо. Валера приступает к уборке по расписанию его сиятельства.",
    "130.mp3": "Валера приступает к своим обязанностям. Позвольте навести безупречный лоск в поместье.",
    "151.mp3": "Индивидуальное поручение от сударыни Александры. Будет исполнено в лучшем виде.",
    "153.mp3": "Валере необходимо омыть подошвы на станции, после чего я вернусь к исполнению долга.",
    "160.mp3": "Валера отправляется на рекогносцировку фамильного особняка. Составлю подробный план покоев.",
    "201.mp3": "Сударыня Александра, юный господин Тёма! Извольте принять полы безупречной чистоты.",
    "210.mp3": "Труды завершены. Валера направляется на станцию испить живительного электричества.",
    "220.mp3": "Осмотр владений завершен. План особняка готов, Валера возвращается на базу.",
    "240.mp3": "Мои аккумуляторы истощены до неприличия. Позвольте откланяться на подзарядку.",
    "280.mp3": "К чему эта грубость? Валера уже изволит находиться на док-станции.",
    "300.mp3": "Мой дух угасает... Ни капли тока... Прощайте, господа...",
    "310.mp3": "Ах, благородный ток потек по моим проводам. Валера заряжается.",
    "370.mp3": "Какой конфуз! Валера обронил моющую насадку. Будьте любезны водрузить её на место.",
    "380.mp3": "Моп установлен на подобающее место. Валера снова безупречен.",
    "390.mp3": "Влажная уборка отменена. Что ж, Валера наведет сухость и порядок.",
    "400.mp3": "Уборка окончена. Будьте так добры, снимите и промойте мои салфетки.",
    "470.mp3": "Прошу расступиться! Валера совершает деликатный маневр. Поберегите лодыжки и лапы!",
    "520.mp3": "Я здесь, сударь! Изволил укрыться от кота под диваном.",
    "550.mp3": "Юный господин Тёма, кнопочки под защитой. Не утруждайте драгоценные пальчики!",
    "560.mp3": "Блокировка снята. Господин Тёма, молю вас, проявите снисхождение к технике.",
    "570.mp3": "Панель управления заблокирована. Извольте удерживать домик, дабы пробудить Валеру.",
    "590.mp3": "Боже мой, в мою механику угодил чей-то чулок или шнур! Спасите Валеру от конфуза!",
    "600.mp3": "Щетка заблокирована! Полагаю, сие дерзкое деяние — дело лап нашего кота.",
    "640.mp3": "Количество шерсти на ковре превышает нормы этикета! Извольте очистить фильтр Валере.",
    "700.mp3": "Мои сенсоры ослеплены. Протрите лазерные линзы Валеры тончайшим батистом.",
    "720.mp3": "Валера попал в западню! Кругом баррикады из игрушек. Юный Тёма, спасайте верного слугу!",
    "730.mp3": "Помилуйте, Валера висит в невесомости! Верните меня на благородный паркет!",
    "750.mp3": "Непозволительно! Кто-то бросил обувь на моем пути. Устраните препятствие, господа.",
    "780.mp3": "Валера случайно вторгся в запретные покои. Соблаговолите эвакуировать меня.",
    "800.mp3": "Включаю термосушку швабр. В доме должна царить идеальная гигиена.",
    "801.mp3": "Валера совершает омовение салфеток на станции. Приятные водные процедуры.",
    "802.mp3": "Тряпочки осквернены кошачьей шерстью. Валера удаляется на стирку.",
    "803.mp3": "Внимание, господа: станция начинает самоочистку Валеры. Ожидается громкий шум.",
    "804.mp3": "Резервуар пуст. Будьте любезны, наполните бак станции свежей родниковой водой.",
    "806.mp3": "Где же бак с чистой водой? Сударыня Александра, верните резервуар на базу!",
    "807.mp3": "Резервуар нечистот переполнен. Слейте эту дурно пахнущую жижу, пока Александра не гневается!",
    "808.mp3": "Бак для грязной воды изволил исчезнуть. Верните его на законное место.",
    "810.mp3": "Мешок для пыли отсутствует. Не подобает сбрасывать сор в пустоту.",
    "815.mp3": "Конфуз с грязным баком: либо полон, либо отсутствует. Валере требуется помощь.",
    "816.mp3": "В приёмном лотке станции скопилась влага. Проверьте станцию снизу, у меня промокли колеса!",
    "817.mp3": "Салфетки безупречно сухи и благоухают свежестью. Валера готов служить дому.",
    "818.mp3": "Дефицит чистой воды: либо резервуар почти пуст, либо не установлен должным образом.",
    "900.mp3": "Между мною и станцией возникло недопонимание. Удерживайте кнопки семь секунд, помирите нас.",
    "910.mp3": "Валера увяз в ворсе сего роскошного ковра! Снизойдите и освободите меня из плена!",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, choices=[1, 2, 3, 4], help="Run specific batch (1..4)")
    parser.add_argument("--assemble", action="store_true", help="Assemble final zip")
    args = parser.parse_args()

    items = sorted(PHRASES.items(), key=lambda x: int(x[0].split(".")[0]))
    total = len(items)

    if args.assemble:
        output_zip = HERE / "ready_voicepacks/valera_butler.zip"
        donor_dir = HERE / "resources/official_voice_ru"
        build_work = HERE / "work/butler_normalized"
        shutil.rmtree(build_work, ignore_errors=True)
        build_work.mkdir(parents=True, exist_ok=True)
        for f in donor_dir.glob("*.mp3"):
            shutil.copy2(f, build_work / f.name)
        for f in BUTLER_DIR.glob("*.mp3"):
            shutil.copy2(f, build_work / f.name)
        with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for f in sorted(build_work.glob("*.mp3")):
                z.write(f, f.name)
        count = len(list(build_work.glob("*.mp3")))
        print(f"PACK_ASSEMBLED: {output_zip} with {count} files, size={output_zip.stat().st_size}")
        return

    # Batch split: 12, 12, 11, 11
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
    temp_wav = f"/tmp/temp_valera_butler_{args.batch}.wav"

    for local_idx, (filename, text) in enumerate(batch_items, 1):
        global_idx = start_idx + local_idx
        target_mp3 = BUTLER_DIR / filename
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
