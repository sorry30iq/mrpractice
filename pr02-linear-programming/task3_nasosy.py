"""
Задача 3. Завод "Хоперский" - топливные (x1) и водяные (x2) насосы.

Расход деталей на 1 насос:
             корпус  платик  манжета  шестерня  прибыль
топливный:     1       4        4        1        50 руб.
водяной:       1       2        4        3       200 руб.

Запас: корпусов 6, платиков 8, манжет 12, шестерней 9.

x1, x2 - целые неотрицательные числа (насосы штучные, дробных быть не может),
поэтому решаем как задачу ЦЕЛОЧИСЛЕННОГО линейного программирования (milp).
"""

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# максимизируем 50*x1 + 200*x2  =>  milp минимизирует, поэтому знак минус
c = np.array([-50, -200])

A = np.array([
    [1, 1],  # корпус
    [4, 2],  # платик
    [4, 4],  # манжета
    [1, 3],  # шестерня
])
b_upper = np.array([6, 8, 12, 9])

constraints = LinearConstraint(A, -np.inf, b_upper)
bounds = Bounds(lb=0, ub=np.inf)
integrality = np.array([1, 1])  # обе переменные - целые

result = milp(c, constraints=constraints, bounds=bounds, integrality=integrality)

print("Статус:", result.message)
print(f"Топливных насосов (x1): {int(round(result.x[0]))}")
print(f"Водяных насосов (x2): {int(round(result.x[1]))}")
print(f"Максимальная прибыль: {-result.fun:.0f} руб.")
