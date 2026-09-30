"""
Задача 4. Биодобавки "Телец" (x1) и "Овен" (x2).

Расход витаминов на 1 кг добавки:
            A     B1    E    прибыль
Телец:      16     8    5     4 тыс.руб
Овен:        4     7    9    7.2 тыс.руб

Запас: A - 784 кг, B1 - 552 кг, E - 567 кг.

Цель: максимизировать прибыль 4*x1 + 7.2*x2.
"""

from scipy.optimize import linprog

c = [-4, -7.2]

A_ub = [
    [16, 4],  # витамин A
    [8, 7],   # витамин B1
    [5, 9],   # витамин E
]
b_ub = [784, 552, 567]
bounds = [(0, None), (0, None)]

result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

print("Статус:", result.message)
print(f"Добавки 'Телец' (x1): {result.x[0]:.2f} кг")
print(f"Добавки 'Овен' (x2): {result.x[1]:.2f} кг")
print(f"Максимальная прибыль: {-result.fun:.2f} тыс. руб.")
