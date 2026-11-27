from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal


#Моторин Михаил
#Настройки
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
CLEAN_DIR = DATASET_DIR / "clean"
NOISY_DIR = DATASET_DIR / "noisy" / "restaurant"

RESULTS_DIR = BASE_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
GRAPHS_DIR = RESULTS_DIR / "graphs"
PROCESSED_DIR = RESULTS_DIR / "processed_audio"

#Три фразы для подробного анализа.
#Можно заменить, например, на ["sp04", "sp12", "sp25"].
SELECTED_PHRASES = ["sp01", "sp02", "sp03"]

SNR_LEVELS = ["0dB", "5dB", "10dB", "15dB"]

#Полосовой фильтр: пропускаем основную область речевого сигнала.
#Для базовой частоты NOIZEUS 8000 Гц диапазон 300–3400 Гц.
FILTER_LOW_HZ = 300.0
FILTER_HIGH_HZ = 3400.0
FILTER_ORDER = 4

#Уровни анализа спектра шума.
FREQUENCY_BANDS = [
    ("0–300 Гц", 0, 300),
    ("300–1000 Гц", 300, 1000),
    ("1000–2000 Гц", 1000, 2000),
    ("2000–3400 Гц", 2000, 3400),
    ("3400–4000 Гц", 3400, 4000),
]

#Вспомогательные методы
#Воронов Арсений

def create_directories():
    """Создаёт папки для результатов."""
    for folder in (TABLES_DIR, GRAPHS_DIR, PROCESSED_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def phrase_id_from_name(path):
    """Ищет в имени файла идентификатор вида sp01."""
    match = re.search(r"(sp\d{2})", path.stem.lower())
    return match.group(1) if match else None


def bit_depth(subtype):
    """Переводит subtype WAV в приблизительную разрядность."""
    mapping = {
        "PCM_U8": 8,
        "PCM_16": 16,
        "PCM_24": 24,
        "PCM_32": 32,
        "FLOAT": 32,
        "DOUBLE": 64,
    }
    return mapping.get(subtype, "не определена")

def amplitude_db(value):
    """Амплитуда в dBFS. Ноль защищён от log10(0)."""
    if value <= 0:
        return -np.inf
    return 20 * np.log10(value)


def read_audio(path):
    """Читает WAV как float64, не меняя исходный файл."""
    audio, sample_rate = sf.read(path, dtype="float64", always_2d=True)
    return audio, sample_rate


def rms_value(audio):
    return float(np.sqrt(np.mean(np.square(audio))))


def calculate_snr(clean, test):
    """
    SNR по формуле из методички:
    10 * log10(sum(clean**2) / sum((clean-test)**2))

    Сравниваем только одинаковое число отсчётов.
    Автоматический сдвиг сигналов не выполняется.
    """
    clean = np.asarray(clean, dtype=np.float64).reshape(-1)
    test = np.asarray(test, dtype=np.float64).reshape(-1)
    if clean.shape != test.shape:
        raise ValueError(
            f"Длины сигналов не совпадают: clean={len(clean)}, test={len(test)}. "
            "SNR нельзя корректно вычислить без проверки временного соответствия."
        )
    count = len(clean)
    if count == 0:
        return np.nan, 0

    clean_part = clean
    test_part = test

    signal_energy = np.sum(clean_part ** 2)
    error_energy = np.sum((clean_part - test_part) ** 2)

    if error_energy == 0:
        return np.inf, count
    if signal_energy == 0:
        return -np.inf, count

    return float(10 * np.log10(signal_energy / error_energy)), count


def find_wav_files(folder):
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() == ".wav"
    )


def build_file_map(folder):
    """Создаёт словарь phrase_id -> путь к WAV."""
    result = {}
    for path in find_wav_files(folder):
        phrase_id = phrase_id_from_name(path)
        if phrase_id:
            #Если в папке встретились дубликаты одного ID,
            #используем первый найденный и выводим предупреждение.
            if phrase_id in result:
                print(f"ПРЕДУПРЕЖДЕНИЕ: несколько файлов для {phrase_id}: "
                      f"{result[phrase_id].name}, {path.name}")
            else:
                result[phrase_id] = path
    return result


#Инвентаризация WAV файлов
#Ветин Костантин

def scan_dataset():
    print("\nЭТАП 1. Проверка WAV-файлов")

    if not CLEAN_DIR.exists():
        raise FileNotFoundError(
            f"Не найдена папка чистых записей: {CLEAN_DIR}"
        )
    if not NOISY_DIR.exists():
        raise FileNotFoundError(
            f"Не найдена папка restaurant: {NOISY_DIR}\n"
            "Проверьте, что download_dataset.py скачал датасет "
            "именно в папку dataset/."
        )

    rows = []
    all_files = find_wav_files(CLEAN_DIR) + find_wav_files(NOISY_DIR)
    if not all_files:
        raise FileNotFoundError(
            "В dataset/ не найдено ни одного WAV-файла. "
            "Сначала выполните: python download_dataset.py"
        )

    for path in all_files:
        try:
            info = sf.info(path)
            audio, sr = read_audio(path)

            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            rms = rms_value(audio) if audio.size else 0.0
            relative = path.relative_to(DATASET_DIR)

            is_clean = CLEAN_DIR in path.parents
            rows.append({
                "file_name": path.name,
                "relative_path": str(relative),
                "record_type": "clean" if is_clean else "noisy",
                "phrase_id": phrase_id_from_name(path),
                "duration_sec": len(audio) / sr if sr else np.nan,
                "sample_rate_Hz": sr,
                "channels": audio.shape[1],
                "bit_depth": bit_depth(info.subtype),
                "subtype": info.subtype,
                "min_amplitude": float(np.min(audio)) if audio.size else np.nan,
                "max_amplitude": float(np.max(audio)) if audio.size else np.nan,
                "peak_amplitude": peak,
                "peak_dBFS": amplitude_db(peak),
                "RMS": rms,
                "RMS_dBFS": amplitude_db(rms),
            })
        except Exception as error:
            print(f"Не удалось прочитать {path}: {error}")

    inventory = pd.DataFrame(rows)
    inventory.to_csv(TABLES_DIR / "inventory.csv", index=False, encoding="utf-8-sig")

    clean_count = int((inventory["record_type"] == "clean").sum()) if not inventory.empty else 0
    noisy_count = int((inventory["record_type"] == "noisy").sum()) if not inventory.empty else 0

    print(f"Найдено чистых WAV: {clean_count}")
    print(f"Найдено WAV с шумом restaurant: {noisy_count}")
    print(f"Всего файлов: {len(inventory)}")
    print(f"Таблица: {TABLES_DIR / 'inventory.csv'}")

    if clean_count != 30 or noisy_count != 120:
        print("ПРЕДУПРЕЖДЕНИЕ: ожидалось 30 clean и 120 noisy файлов.")
        print("Проверьте полноту загрузки датасета.")

    return inventory

#Проверка соответствия фраз

def check_pairs():
    print("\nПроверка соответствия clean/noisy")

    clean_map = build_file_map(CLEAN_DIR)
    rows = []

    for level in SNR_LEVELS:
        level_dir = NOISY_DIR / level
        noisy_map = build_file_map(level_dir)

        for phrase_id in sorted(set(clean_map) | set(noisy_map)):
            rows.append({
                "phrase_id": phrase_id,
                "snr_folder": level,
                "clean_exists": phrase_id in clean_map,
                "noisy_exists": phrase_id in noisy_map,
                "clean_file": clean_map.get(phrase_id, ""),
                "noisy_file": noisy_map.get(phrase_id, ""),
            })

    pairs = pd.DataFrame(rows)
    pairs.to_csv(TABLES_DIR / "pair_check.csv", index=False, encoding="utf-8-sig")

    missing = pairs[(~pairs["clean_exists"]) | (~pairs["noisy_exists"])]
    print(f"Проверено сочетаний фраза/уровень: {len(pairs)}")
    print(f"Отсутствующих пар: {len(missing)}")

    if len(missing):
        print(missing[["phrase_id", "snr_folder", "clean_exists", "noisy_exists"]])
    return clean_map




#Графики: WAVEFORM, FFT, SPECTROGRAM
#Васильев Андрей

def spectrum_db(audio, sample_rate):
    """Односторонний спектр амплитуды в dB."""
    mono = np.mean(audio, axis=1)
    if len(mono) == 0:
        return np.array([0.0]), np.array([-120.0])
    window = np.hanning(len(mono))
    spectrum = np.abs(np.fft.rfft(mono * window))
    spectrum = spectrum / max(np.sum(window) / 2, 1e-12)
    frequencies = np.fft.rfftfreq(len(mono), d=1 / sample_rate)
    spectrum_db_values = 20 * np.log10(np.maximum(spectrum, 1e-10))
    return frequencies, spectrum_db_values


def plot_phrase_analysis(phrase_id, clean_path, noisy_paths):
    """Один рисунок на фразу: строки — clean и 4 уровня шума."""
    records = [("Чистая запись", clean_path)]
    for level in SNR_LEVELS:
        records.append((f"Шум {level}", noisy_paths[level]))

    loaded = []
    for label, path in records:
        audio, sr = read_audio(path)
        loaded.append((label, audio, sr))

    sample_rates = {item[2] for item in loaded}
    if len(sample_rates) != 1:
        print(f"Пропуск графиков {phrase_id}: разные sample rate.")
        return

    sr = loaded[0][2]
    fig, axes = plt.subplots(5, 3, figsize=(17, 16))
    fig.suptitle(
        f"{phrase_id}: ресторанный шум — временная и частотная области",
        fontsize=15
    )

    #Единый предел FFT для всех пяти сигналов.
    all_spectra = [
        spectrum_db(audio, sr)[1] for _, audio, _ in loaded
    ]
    fft_max = max(float(np.max(x)) for x in all_spectra)
    fft_min = max(-120.0, min(float(np.min(x)) for x in all_spectra))

    for row, (label, audio, sample_rate) in enumerate(loaded):
        mono = np.mean(audio, axis=1)
        if len(mono) == 0:
            print(f"Пропуск графиков {phrase_id}: найден пустой аудиофайл.")
            plt.close(fig)
            return
        time_axis = np.arange(len(mono)) / sample_rate

        #Waveform
        axes[row, 0].plot(time_axis, mono, linewidth=0.45)
        axes[row, 0].set_xlim(0, max(time_axis[-1], 0.001))
        axes[row, 0].set_ylim(-1.0, 1.0)
        axes[row, 0].set_ylabel(f"{label}\nАмплитуда, FS")
        axes[row, 0].set_xlabel("Время, с")
        axes[row, 0].grid(True, alpha=0.25)

        #FFT
        frequencies, values_db = spectrum_db(audio, sample_rate)
        axes[row, 1].plot(frequencies, values_db, linewidth=0.55)
        axes[row, 1].set_xlim(0, sample_rate / 2)
        axes[row, 1].set_ylim(fft_min, fft_max + 3)
        axes[row, 1].set_xlabel("Частота, Гц")
        axes[row, 1].set_ylabel("Амплитуда, дБ")
        axes[row, 1].grid(True, alpha=0.25)

        #Spectrogram
        nperseg = min(256, len(mono))
        if nperseg >= 16:
            frequencies_s, times_s, power = signal.spectrogram(
                mono, fs=sample_rate, nperseg=nperseg,
                noverlap=nperseg // 2, scaling="density"
            )
            power_db = 10 * np.log10(np.maximum(power, 1e-14))
            axes[row, 2].pcolormesh(
                times_s, frequencies_s, power_db,
                shading="auto", vmin=-100, vmax=0
            )
            axes[row, 2].set_ylim(0, sample_rate / 2)
        axes[row, 2].set_xlabel("Время, с")
        axes[row, 2].set_ylabel("Частота, Гц")

    axes[0, 0].set_title("Временная форма (waveform)")
    axes[0, 1].set_title("Амплитудный спектр FFT")
    axes[0, 2].set_title("Спектрограмма (PSD, дБ/Гц)")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output = GRAPHS_DIR / f"{phrase_id}_analysis.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"График сохранён: {output}")


#Анализ шума и SNR
#Воронов Арсений и Васильев Андрей

def mono_signal(audio):
    return np.mean(audio, axis=1)


def noise_frequency_table(clean, noisy, sample_rate, phrase_id, level):
    """
    Спектр ошибки noisy-clean используется как оценка шума.
    Это приближение: в корпусе шум может быть нестационарным.
    """
    if clean.shape != noisy.shape:
        return []
    count = len(clean)
    if count == 0:
        return []

    residual = mono_signal(noisy - clean)
    frequencies, psd = signal.welch(
        residual, fs=sample_rate, nperseg=min(1024, count)
    )
    rows = []

    for band_name, low, high in FREQUENCY_BANDS:
        actual_high = min(high, sample_rate / 2)
        mask = (frequencies >= low) & (frequencies < actual_high)
        if np.any(mask):
            power = float(np.trapezoid(psd[mask], frequencies[mask]))
        else:
            power = 0.0

        rows.append({
            "phrase_id": phrase_id,
            "snr_folder": level,
            "frequency_band": band_name,
            "noise_power": power,
            "noise_power_dB": 10 * np.log10(max(power, 1e-20)),
        })

    return rows


def analyze_all_snr(clean_map):
    print("\nЭТАП 3. Измерение SNR и анализ спектра шума")

    snr_rows = []
    frequency_rows = []

    for level in SNR_LEVELS:
        noisy_map = build_file_map(NOISY_DIR / level)

        for phrase_id, noisy_path in sorted(noisy_map.items()):
            clean_path = clean_map.get(phrase_id)
            if clean_path is None:
                continue

            clean, sr_clean = read_audio(clean_path)
            noisy, sr_noisy = read_audio(noisy_path)

            if sr_clean != sr_noisy:
                print(f"Пропуск SNR {phrase_id}/{level}: разные частоты дискретизации.")
                continue
            if clean.shape != noisy.shape:
                print(
                    f"Пропуск SNR {phrase_id}/{level}: форма сигналов не совпадает "
                    f"({clean.shape} и {noisy.shape})."
                )
                continue

            snr, count = calculate_snr(mono_signal(clean), mono_signal(noisy))
            snr_rows.append({
                "phrase_id": phrase_id,
                "snr_folder": level,
                "clean_file": clean_path.name,
                "noisy_file": noisy_path.name,
                "sample_rate_Hz": sr_clean,
                "compared_samples": count,
                "compared_duration_sec": count / sr_clean,
                "measured_SNR_dB": snr,
            })

            frequency_rows.extend(
                noise_frequency_table(clean, noisy, sr_clean, phrase_id, level)
            )

    snr_df = pd.DataFrame(snr_rows)
    snr_df.to_csv(TABLES_DIR / "snr_all_noisy.csv", index=False, encoding="utf-8-sig")

    freq_df = pd.DataFrame(frequency_rows)
    freq_df.to_csv(
        TABLES_DIR / "noise_frequency_bands.csv",
        index=False, encoding="utf-8-sig"
    )

    if not snr_df.empty:
        summary = (
            snr_df.groupby("snr_folder")["measured_SNR_dB"]
            .agg(["count", "mean", "std", "min", "max"])
            .reset_index()
        )
        summary.to_csv(
            TABLES_DIR / "snr_by_folder_summary.csv",
            index=False, encoding="utf-8-sig"
        )
        print("\nИзмеренный SNR по папкам (не обязан совпадать с номиналом):")
        print(summary.to_string(index=False))

    if not freq_df.empty:
        band_summary = (
            freq_df.groupby("frequency_band")["noise_power_dB"]
            .mean().reset_index()
        )
        band_summary.to_csv(
            TABLES_DIR / "noise_frequency_summary.csv",
            index=False, encoding="utf-8-sig"
        )
        print("\nСредняя мощность остаточного шума по частотным полосам:")
        print(band_summary.to_string(index=False))

    return snr_df


#Полосовая фильтрация
#Филиповский Виталий

def design_bandpass(sample_rate):
    """
    Проектирует один полосовой фильтр Баттерворта в форме SOS.
    Верхняя частота автоматически ограничивается частотой Найквиста.
    """
    nyquist = sample_rate / 2
    low = FILTER_LOW_HZ
    high = min(FILTER_HIGH_HZ, nyquist * 0.95)

    if low >= high:
        raise ValueError(
            f"Невозможно создать полосовой фильтр: {low}–{high} Гц "
            f"при sample rate {sample_rate} Гц."
        )

    sos = signal.butter(
        FILTER_ORDER,
        [low, high],
        btype="bandpass",
        fs=sample_rate,
        output="sos"
    )
    return sos, low, high


def apply_filter(audio, sample_rate):
    """Фильтрует каналы отдельно, не меняя количество отсчётов."""
    sos, low, high = design_bandpass(sample_rate)

    if len(audio) < 32:
        raise ValueError("Слишком короткая запись для фильтрации.")

    filtered = np.zeros_like(audio)
    for channel in range(audio.shape[1]):
        filtered[:, channel] = signal.sosfiltfilt(sos, audio[:, channel])

    return filtered, low, high


#Эксперементы
#Тагильцева Дарья, Моторин Михаил и Ветин Константин

def plot_before_after_spectrum(phrase_id, level, clean, noisy, filtered, sr):
    """Сравнивает FFT чистого, шумного и отфильтрованного сигналов."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for label, audio in (
        ("Чистый эталон", clean),
        ("До фильтрации", noisy),
        ("После фильтрации", filtered),
    ):
        freq, values = spectrum_db(audio, sr)
        ax.plot(freq, values, linewidth=0.8, label=label)

    ax.set_xlim(0, sr / 2)
    ax.set_xlabel("Частота, Гц")
    ax.set_ylabel("Амплитуда спектра, дБ")
    ax.set_title(f"{phrase_id}, {level}: сравнение спектров")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output = GRAPHS_DIR / f"{phrase_id}_{level}_before_after_fft.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)


def run_experiments(clean_map):
    print("\nЭТАП 4–5. Фильтрация выбранных 12 записей и оценка")

    experiment_rows = []
    processed_count = 0

    for phrase_id in SELECTED_PHRASES:
        clean_path = clean_map.get(phrase_id)
        if clean_path is None:
            print(f"ПРЕДУПРЕЖДЕНИЕ: нет чистого файла {phrase_id}.")
            continue

        clean, sr_clean = read_audio(clean_path)

        for level in SNR_LEVELS:
            noisy_map = build_file_map(NOISY_DIR / level)
            noisy_path = noisy_map.get(phrase_id)

            if noisy_path is None:
                print(f"ПРЕДУПРЕЖДЕНИЕ: нет {phrase_id} в {level}.")
                continue

            noisy, sr_noisy = read_audio(noisy_path)

            if sr_clean != sr_noisy:
                print(f"Пропуск {phrase_id}/{level}: sample rate не совпадает.")
                continue
            if clean.shape != noisy.shape:
                print(
                    f"Пропуск {phrase_id}/{level}: длина/число каналов не совпадает "
                    f"({clean.shape} и {noisy.shape})."
                )
                continue

            filtered, low, high = apply_filter(noisy, sr_noisy)

            #Фильтр не должен изменять число отсчётов.
            if len(filtered) != len(noisy):
                raise RuntimeError("После фильтрации изменилась длина аудио.")

            output_name = f"{phrase_id}_restaurant_{level}_bandpass.wav"
            output_path = PROCESSED_DIR / output_name

            #Сохраняем обработанную копию, исходный WAV не изменяется.
            sf.write(output_path, filtered, sr_noisy, subtype="PCM_16")
            processed_count += 1

            clean_mono = mono_signal(clean)
            noisy_mono = mono_signal(noisy)
            filtered_mono = mono_signal(filtered)

            snr_before, samples_before = calculate_snr(clean_mono, noisy_mono)
            snr_after, samples_after = calculate_snr(clean_mono, filtered_mono)
            delta_snr = snr_after - snr_before

            peak_before = float(np.max(np.abs(noisy)))
            peak_after = float(np.max(np.abs(filtered)))
            clipping_after = peak_after >= 1.0

            if delta_snr > 0.1:
                comment = "SNR увеличился: улучшение по выбранной метрике."
            elif delta_snr < -0.1:
                comment = "SNR уменьшился: фильтр ухудшил сходство с эталоном."
            else:
                comment = "Изменение SNR небольшое."

            if clipping_after:
                comment += " Обнаружен возможный клиппинг."

            common_samples = min(samples_before, samples_after)
            experiment_rows.append({
                "phrase_id": phrase_id,
                "snr_folder": level,
                "stage": "до фильтрации",
                "filter": "Полосовой Баттерворт",
                "filter_order": FILTER_ORDER,
                "low_cut_Hz": low,
                "high_cut_Hz": high,
                "sample_rate_Hz": sr_noisy,
                "compared_samples": samples_before,
                "SNR_dB": snr_before,
                "delta_SNR_dB": 0.0,
                "peak_amplitude": peak_before,
                "clipping": peak_before >= 1.0,
                "comment": "Исходная зашумлённая запись.",
                "output_file": noisy_path.name,
            })
            experiment_rows.append({
                "phrase_id": phrase_id,
                "snr_folder": level,
                "stage": "после фильтрации",
                "filter": "Полосовой Баттерворт",
                "filter_order": FILTER_ORDER,
                "low_cut_Hz": low,
                "high_cut_Hz": high,
                "sample_rate_Hz": sr_noisy,
                "compared_samples": common_samples,
                "SNR_dB": snr_after,
                "delta_SNR_dB": delta_snr,
                "peak_amplitude": peak_after,
                "clipping": clipping_after,
                "comment": comment,
                "output_file": output_name,
            })

            plot_before_after_spectrum(
                phrase_id, level, clean, noisy, filtered, sr_noisy
            )

            print(
                f"{phrase_id} / {level}: "
                f"SNR до = {snr_before:.2f} дБ; "
                f"после = {snr_after:.2f} дБ; "
                f"ΔSNR = {delta_snr:.2f} дБ"
            )

    experiments = pd.DataFrame(experiment_rows)
    experiments.to_csv(
        TABLES_DIR / "experiments_before_after.csv",
        index=False, encoding="utf-8-sig"
    )

    print(f"\nСохранено обработанных WAV: {processed_count}")
    if processed_count < 12:
        print("ПРЕДУПРЕЖДЕНИЕ: получилось меньше 12 WAV. "
              "Проверьте наличие всех выбранных файлов.")

    return experiments


#Резюме текстом
#Тагильцева Дарья

def write_summary(inventory, snr_df, experiments):
    output = RESULTS_DIR / "summary.txt"

    lines = [
        "Лабораторная работа № 3 — вариант 04 (restaurant)",
        "",
        "Метод обработки: полосовой фильтр Баттерворта.",
        f"Порядок фильтра: {FILTER_ORDER}.",
        f"Нижняя частота среза: {FILTER_LOW_HZ} Гц.",
        f"Верхняя частота среза: {FILTER_HIGH_HZ} Гц "
        "(может быть ограничена частотой Найквиста).",
        f"Выбранные фразы: {', '.join(SELECTED_PHRASES)}.",
        "",
        "Фильтр пропускает область, в которой сосредоточена значительная "
        "часть речевой информации, и ослабляет низкочастотные и "
        "высокочастотные составляющие за её пределами.",
        "Ресторанный шум может включать речь других людей, звон посуды "
        "и фоновые звуки. Их частоты частично совпадают с частотами "
        "полезной речи, поэтому один полосовой фильтр не способен "
        "полностью удалить такие помехи.",
        "",
        "Численные результаты необходимо интерпретировать по таблицам "
        "и графикам. Положительное ΔSNR означает улучшение сходства "
        "с чистым эталоном по формуле SNR; отрицательное — ухудшение.",
        "SNR, измеренный по полной записи, может отличаться от "
        "номинального значения в названии папки корпуса.",
        "",
        "Файлы результатов:",
        "- tables/inventory.csv",
        "- tables/pair_check.csv",
        "- tables/snr_all_noisy.csv",
        "- tables/snr_by_folder_summary.csv",
        "- tables/noise_frequency_bands.csv",
        "- tables/noise_frequency_summary.csv",
        "- tables/experiments_before_after.csv",
        "- graphs/",
        "- processed_audio/",
    ]

    if not inventory.empty:
        lines += [
            "",
            f"Всего просканировано WAV: {len(inventory)}.",
            f"Чистых: {(inventory['record_type'] == 'clean').sum()}.",
            f"Зашумлённых: {(inventory['record_type'] == 'noisy').sum()}.",
        ]

    if snr_df is not None and not snr_df.empty:
        lines += ["", "Средний измеренный SNR по папкам:"]
        summary = snr_df.groupby("snr_folder")["measured_SNR_dB"].mean()
        for level, value in summary.items():
            lines.append(f"- {level}: {value:.2f} дБ")

    if experiments is not None and not experiments.empty:
        after = experiments[experiments["stage"] == "после фильтрации"]
        lines += [
            "",
            f"Среднее изменение SNR после фильтрации: "
            f"{after['delta_SNR_dB'].mean():.2f} дБ.",
            f"Количество записей с положительным ΔSNR: "
            f"{(after['delta_SNR_dB'] > 0).sum()} из {len(after)}.",
            "",
            "Вывод для отчёта: проверьте отдельные значения ΔSNR, "
            "прослушайте исходные и обработанные WAV и сопоставьте "
            "результат с графиками. Не утверждайте, что фильтр улучшил "
            "все записи, если таблица этого не подтверждает.",
        ]

    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Резюме сохранено: {output}")

#Мейн
#Васильев Андрей

def main():
    create_directories()

    print("ЛАБОРАТОРНАЯ РАБОТА № 3 — ВАРИАНТ 04: РЕСТОРАН")

    inventory = scan_dataset()
    clean_map = check_pairs()

    #Подробные графики для выбранных трёх фраз.
    for phrase_id in SELECTED_PHRASES:
        clean_path = clean_map.get(phrase_id)
        noisy_map = build_file_map(NOISY_DIR / SNR_LEVELS[0])

        if clean_path is None:
            print(f"Нет чистой записи {phrase_id}, график пропущен.")
            continue

        selected_noisy = {}
        all_levels_exist = True
        for level in SNR_LEVELS:
            level_map = build_file_map(NOISY_DIR / level)
            if phrase_id not in level_map:
                all_levels_exist = False
                print(f"Нет {phrase_id} в {level}.")
                break
            selected_noisy[level] = level_map[phrase_id]

        if all_levels_exist:
            plot_phrase_analysis(phrase_id, clean_path, selected_noisy)

    snr_df = analyze_all_snr(clean_map)
    experiments = run_experiments(clean_map)
    write_summary(inventory, snr_df, experiments)

    print("\nГОТОВО.")
    print(f"Все результаты находятся в папке: {RESULTS_DIR}")
    print("Оригинальные WAV не изменялись.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nРабота остановлена пользователем.")
        sys.exit(1)
    except Exception as error:
        print(f"\nОШИБКА: {error}")
        print("Проверьте структуру dataset/ и установленные библиотеки.")
        raise
