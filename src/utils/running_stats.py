"""
Класс для вычисления среднего и дисперсии данных без необходимости
загружать все значения в память сразу.
Реализует алгоритм Уэлфорда для устойчивого онлайн-вычисления статистик.
Источник: http://www.johndcook.com/standard_deviation.html
"""

import math
from typing import Optional

class RunningStats:
    """
    Онлайн-калькулятор среднего, дисперсии и стандартного отклонения.

    Атрибуты:
        n (int): Количество обработанных значений.
        old_m (float): Предыдущее значение среднего.
        new_m (float): Текущее значение среднего.
        old_s (float): Предыдущее значение суммы квадратов отклонений.
        new_s (float): Текущее значение суммы квадратов отклонений.
        min_val (Optional[float]): Минимальное значение.
        max_val (Optional[float]): Максимальное значение.
    """

    def __init__(self) -> None:
        """
        Инициализация объекта RunningStats.
        """
        self.n: int = 0
        self.old_m: float = 0.0
        self.new_m: float = 0.0
        self.old_s: float = 0.0
        self.new_s: float = 0.0
        self.min_val: Optional[float] = None
        self.max_val: Optional[float] = None

    def clear(self) -> None:
        """
        Сброс всех статистик.
        """
        self.n = 0
        self.old_m = 0.0
        self.new_m = 0.0
        self.old_s = 0.0
        self.new_s = 0.0
        self.min_val = None
        self.max_val = None

    def push(self, x: float) -> None:
        """
        Добавить новое значение и обновить статистики.

        Аргументы:
            x (float): Новое значение.
        """
        self.n += 1

        # Обновление минимального и максимального значения
        if self.min_val is None or x < self.min_val:
            self.min_val = x
        if self.max_val is None or x > self.max_val:
            self.max_val = x

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0.0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)
            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self) -> float:
        """
        Получить текущее среднее значение.

        Возвращает:
            float: Среднее значение.
        """
        if self.n == 0:
            return 0.0
        assert self.min_val is not None and self.max_val is not None
        assert self.min_val <= self.new_m <= self.max_val
        return self.new_m

    def variance(self) -> float:
        """
        Получить текущую выборочную дисперсию.

        Возвращает:
            float: Дисперсия.
        """
        if self.n <= 1:
            return 0.0
        assert self.min_val is not None and self.max_val is not None
        var2 = self.new_s / (self.n - 1)
        m = max(abs(self.max_val), abs(self.min_val))
        assert var2 <= m * m
        return var2

    def standard_deviation(self) -> float:
        """
        Получить текущее стандартное отклонение.

        Возвращает:
            float: Стандартное отклонение.
        """
        return math.sqrt(self.variance())