# Antibot_Avito

Итоговое решение и все пояснения находятся в [solution.ipynb](solution.ipynb).

1. Использовать Python 3.13.7 и установить зависимости: `pip install -r requirements.txt`.
2. Положить исходные `train.csv`, `test.csv`, `events.csv.gz` в `data/`; рядом с ноутбуком нужны `metric.py` и `sample_submission.csv`.
3. Выполнить **Restart Kernel → Run All** в `solution.ipynb`. Он создаст `submission.csv` с колонками `cookie_id,score`.

Модель: CatBoost, агрегаты событий только внутри окна cookie, вариант «Паузы 1–99%».
Валидация: два последовательных временных фолда; отбор по среднему максимальному Precision при Recall ≥ 70% на seed 42, 43, 44.
Локальная метрика выбранного варианта: 0.7415; независимого holdout нет.
Признаки, baseline, эксперименты, ошибки, ограничения и источники описаны в ноутбуке.
Использованы NumPy, pandas, scikit-learn, CatBoost и matplotlib; официальная метрика и baseline — из задания.
`quickstart.ipynb` сохранён как исходный пример; точка входа для финальной сдачи — `solution.ipynb`.
