# Antibot_Avito

Итоговое решение и пояснения — в [solution.ipynb](solution.ipynb).

1. Python 3.13.7; установить зависимости: `pip install -r requirements.txt`.
2. Исходные `train.csv`, `test.csv`, `events.csv.gz` должны находиться в `data/`; рядом нужны `metric.py` и `sample_submission.csv`.
3. Выполнить **Restart Kernel → Run All**. Ноутбук создаст итоговый `submission.csv` с колонками `cookie_id,score`.

Итог: CatBoost с параметрами Optuna, 69 признаков «Паузы 1–99%», 952 дерева, seed 42.
Сообщённая оценка скрытого теста: **0,79**; исходная модель получила 0,78.

**Устойчивость:** на двух временных фолдах и трёх seed средняя метрика кандидата — 0,7299
против 0,7415 у исходных параметров, одна победа из шести. Локальное преимущество не подтверждено;
фолды и обратная связь скрытого теста использовались при отборе, независимого holdout нет.

Признаки, baseline, Optuna-поиск (20 попыток), ошибки и ограничения описаны в ноутбуке.
Использованы NumPy, pandas, scikit-learn, CatBoost, Optuna и matplotlib;
официальная метрика и baseline взяты из задания. Версии — в `requirements.txt`.
`quickstart.ipynb` сохранён как исходный пример; точка входа для сдачи — `solution.ipynb`.
