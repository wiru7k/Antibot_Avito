"""
Официальная метрика кейса: Precision @ Recall >= 0.70.

Этот же файл использует проверяющая система. Считайте им метрику на своей валидации,
чтобы локальные числа совпадали с итоговыми.

Определение
-----------
Куки сортируются по `score` по убыванию. Перебираются все пороги; для каждого
отмечаются куки со `score >= threshold` и считаются precision и recall.
Результат — **максимальный precision среди порогов, у которых recall >= 0.70**.

Две детали, в которых легко ошибиться:

1. Одинаковые score обрабатываются **одной группой**. Нельзя разделить куки с равным
   score: либо отмечены все, либо никто. Порядок строк на результат не влияет.
2. Берётся именно максимум по допустимой области, а не первая точка, где полнота
   достигнута. Precision вдоль кривой не монотонен, и эти два числа различаются.
   Пример: ответы в порядке убывания score — 1, 0, 1, 1, 1. Первая точка с recall >= 0.7
   даёт precision 0.75, следующая — 0.80. Правильный ответ 0.80.
"""

from __future__ import annotations

import numpy as np

TARGET_RECALL = 0.70


def pr_curve(y_true, score) -> tuple[np.ndarray, np.ndarray]:
    """(precision, recall) в точках на границах групп одинакового score."""
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    if y_true.shape != score.shape:
        raise ValueError("y_true и score разной длины")

    n_pos = int(y_true.sum())
    if n_pos == 0:
        return np.array([]), np.array([])

    order = np.argsort(-score, kind="mergesort")
    y, s = y_true[order], score[order]

    tp = np.cumsum(y)
    k = np.arange(1, len(y) + 1)
    ends = np.r_[s[1:] != s[:-1], True]      # последняя строка каждой группы равных score
    return tp[ends] / k[ends], tp[ends] / n_pos


def precision_at_recall(y_true, score, recall: float = TARGET_RECALL) -> float:
    """Максимальный precision среди порогов с recall >= `recall`."""
    prec, rec = pr_curve(y_true, score)
    if len(prec) == 0:
        return float("nan")
    ok = rec >= recall
    return float(prec[ok].max()) if ok.any() else 0.0


def recall_at_fpr(y_true, score, fpr: float = 0.01) -> float:
    """Диагностика: какую долю ботов ловим, задев не более `fpr` доли людей.

    Как и основная метрика, работает по группам одинакового score.
    """
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    n_pos, n_neg = int(y_true.sum()), int((1 - y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(-score, kind="mergesort")
    y, s = y_true[order], score[order]
    ends = np.r_[s[1:] != s[:-1], True]
    tp = np.cumsum(y)[ends]
    fp = np.cumsum(1 - y)[ends]

    allowed = fp <= fpr * n_neg
    return float(tp[allowed].max() / n_pos) if allowed.any() else 0.0


if __name__ == "__main__":
    # самопроверка на примерах из описания
    assert precision_at_recall([1, 0, 1, 1, 1], [5, 4, 3, 2, 1]) == 0.8
    assert precision_at_recall([1, 1, 0, 0], [0.5] * 4) == 0.5      # группа равных score
    assert precision_at_recall([0, 0, 1, 1], [0.5] * 4) == 0.5      # порядок не влияет
    assert recall_at_fpr(np.r_[np.ones(10), np.zeros(90)], np.full(100, 0.5)) == 0.0
    print("метрика: самопроверка пройдена")
