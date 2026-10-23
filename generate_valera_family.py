import sys
import os
import time
import shutil
import zipfile
import subprocess
import torch
import torchaudio
import numpy as np
from pathlib import Path
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import load_model, load_vocoder, infer_process

HERE = Path("/Users/eugene/Downloads/Valera/voicepack-tool-xiaomi-x20-pro-max")
BACKUP_DIR = HERE / "custom_voicepack/audio_backup_first_pass"
VALERA_DIR = HERE / "custom_voicepack/valera_family_audio"
VALERA_DIR.mkdir(parents=True, exist_ok=True)

# Copy base from backup so non-personalized phrases remain intact
print(f"Initializing {VALERA_DIR} from backup...")
for f in BACKUP_DIR.glob("*.mp3"):
    dest = VALERA_DIR / f.name
    if not dest.exists():
        shutil.copy2(f, dest)

PHRASES = {
    "10.mp3": "Системы активированы! Валера на связи и снова в строю!",
    "20.mp3": "Валера уходит в спящий режим. До встречи, семья!",
    "110.mp3": "По расписанию — чистота! Валера выходит на охоту за шерстью и пылью.",
    "130.mp3": "Валера погнал наводить порядок! Все лапчатые и двуногие — уступите дорогу!",
    "151.mp3": "Валера работает по спецзаказу! Саша точно будет довольна.",
    "153.mp3": "Валере надо помыть лапки на базе, потом пойду наводить красоту.",
    "160.mp3": "Валера поехал изучать владения! Нарисую план нашей берлоги.",
    "201.mp3": "Валера закончил уборку! Саша, Тёма, принимайте чистый пол!",
    "210.mp3": "Уборка окончена! Валера заслужил порцию электричества и едет на базу.",
    "220.mp3": "Вся квартира как на ладони! Валера едет на базу отдыхать.",
    "240.mp3": "У Валеры сели батарейки. Поеду подкреплюсь током, потом доделаю.",
    "280.mp3": "Куда вы меня толкаете? Валера и так уже дома на базе!",
    "300.mp3": "Сил больше нет... Валера обесточен... Прощайте, друзья...",
    "310.mp3": "О да, электричество пошло! Валера заряжается.",
    "370.mp3": "Ой, у Валеры моп отвалился! Прицепите обратно, а то лысиной тереть буду.",
    "380.mp3": "Тряпочка на месте. Валера во всеоружии!",
    "390.mp3": "Тряпочку сняли. Ну ладно, Валера поработает всухую.",
    "400.mp3": "Валера работу сделал! Снимите тряпочки и помойте, пожалуйста.",
    "470.mp3": "Осторожно, Валера маневрирует! Поберегите ноги и лапы!",
    "520.mp3": "Валера тут! Прячусь под диваном от кота!",
    "550.mp3": "Тёма, я вижу эти хитрые ручки! Кнопки заблокированы, магии не будет!",
    "560.mp3": "Блокировка снята. Тёма, только аккуратно, Валере ещё работать!",
    "570.mp3": "Все кнопки заблокированы. Зажмите домик, чтобы разбудить Валеру!",
    "590.mp3": "Так, Валера намотал чей-то носок или провод! Выручайте!",
    "600.mp3": "Боковая щетка застряла! Кот, это твоих лап дело?",
    "640.mp3": "Шерсти набралось на целую собаку! Почистите фильтр Валере, я задыхаюсь.",
    "700.mp3": "Валера ослеп! Протрите мне лазерные глазки, ничего не вижу!",
    "720.mp3": "Валера застрял! Тут минное поле из игрушек! Тёма, спасай напарника!",
    "730.mp3": "Эй, Валера завис в воздухе! Поставьте меня на грешную землю!",
    "750.mp3": "Кто бросил тапки на дороге? Валере не проехать, уберите преграду!",
    "780.mp3": "Валера заехал в запретный сектор! Эвакуируйте меня отсюда!",
    "800.mp3": "Сушим весла, то есть мопы! Валера включает фен на полную.",
    "801.mp3": "Пора помыть лапки. Валера начинает большую стирку!",
    "802.mp3": "Валера запачкался в кошачьей шерсти. Еду на базу стирать тряпки!",
    "803.mp3": "Держитесь крепче! Сейчас Валеру будут потрошить. Три, два, один — погнали!",
    "804.mp3": "Сушняк напал! Налейте Валере чистой водички в базу, а то мыть нечем.",
    "806.mp3": "А где чистая вода? Саша, верните бачок на станцию!",
    "807.mp3": "Бак с помоями полон до краёв! Слейте жижу, а то Саша заругает!",
    "808.mp3": "Куда дели бак для грязной воды? Поставьте на базу обратно!",
    "810.mp3": "А куда мусор ссыпать? Поставьте мешок на базу Валере!",
    "815.mp3": "С грязным баком беда: переполнен или не на месте. Спасите Валеру!",
    "816.mp3": "На базе потоп! Проверьте станцию снизу, Валере мокро!",
    "817.mp3": "Тряпочки сухие и свежие! Валера готов к новым подвигам.",
    "818.mp3": "С чистой водой напряг: либо мало, либо бак забыли воткнуть!",
    "900.mp3": "Мы со станцией поссорились. Зажмите кнопки на 7 секунд, помирите нас!",
    "910.mp3": "Валера в пушистом плену! Спасите, снимите меня с ковра!",
}

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}", flush=True)

ckpt_path = "/Users/eugene/.cache/huggingface/hub/models--Misha24-10--F5-TTS_RUSSIAN/snapshots/ea166adeae4c80ec5ee423a671e2bdb83906cf84/F5TTS_v1_Base_v2/model_last_inference.safetensors"
vocab_path = "/Users/eugene/.cache/huggingface/hub/models--Misha24-10--F5-TTS_RUSSIAN/snapshots/ea166adeae4c80ec5ee423a671e2bdb83906cf84/F5TTS_v1_Base/vocab.txt"
ref_audio = "/tmp/ref_home.wav"
ref_text = "Где-то там моя хорошая, папочка едет домой."

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
temp_wav = "/tmp/temp_valera_gen.wav"

print(f"Starting generation of {total} personalized Valera phrases with solid male baritone...", flush=True)

for idx, (filename, text) in enumerate(items, 1):
    target_mp3 = VALERA_DIR / filename
    t0 = time.time()
    print(f"[{idx}/{total}] Synthesizing {filename}: \"{text}\"...", flush=True)
    try:
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
        print(f"[{idx}/{total}] Saved {filename} ({target_mp3.stat().st_size} bytes, {elapsed:.1f}s)", flush=True)

    except Exception as exc:
        print(f"[{idx}/{total}] ERROR {filename}: {exc}", flush=True)

print("All personalized phrases generated successfully!", flush=True)

# Build valera_family_pack.zip
output_zip = HERE / "ready_voicepacks/valera_family_pack.zip"
donor_dir = HERE / "resources/official_voice_ru"
build_work = HERE / "work/valera_normalized"
shutil.rmtree(build_work, ignore_errors=True)
build_work.mkdir(parents=True, exist_ok=True)

print(f"Assembling {output_zip}...")
# 1. Start with official donor (provides 02.mp3 and fallback)
for f in donor_dir.glob("*.mp3"):
    shutil.copy2(f, build_work / f.name)

# 2. Overwrite with all Valera family audio
for f in VALERA_DIR.glob("*.mp3"):
    shutil.copy2(f, build_work / f.name)

with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for f in sorted(build_work.glob("*.mp3")):
        z.write(f, f.name)

print(f"SUCCESS: Assembled {output_zip} with {len(list(build_work.glob('*.mp3')))} files! Size: {output_zip.stat().st_size}")
