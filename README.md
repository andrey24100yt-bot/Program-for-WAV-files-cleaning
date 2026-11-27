# Program-for-WAV-files-cleaning
Программа для отчистки звуковых файлов из датасетов. Проверено на 04_Ресторан.

Скрипт:
1. Проверяет все WAV в dataset/ и сохраняет inventory.csv.
2. Проверяет пары clean/noisy по идентификаторам sp01...sp30.
3. Для трёх выбранных фраз строит waveform, FFT и спектрограммы.
4. Измеряет SNR исходных записей и характеристики спектра шума.
5. Применяет один частотный фильтр: полосовой фильтр Баттерворта.
6. Сохраняет 12 обработанных WAV и таблицу экспериментов (24 строки).

Ожидаемая структура:
project/
    main.py
    download_dataset.py
    dataset/
        clean/
        noisy/
            restaurant/
                0dB/
                5dB/
                10dB/
                15dB/
    results/  (создаётся автоматически)

Перед запуском:
    python -m pip install -r requirements.txt

Сначала скачайте данные: python download_dataset.py
Затем запустите обработку: python main.py
