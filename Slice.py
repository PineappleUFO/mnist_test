import pandas as pd
import numpy as np

# Использую библиотеку sklearn для генерации более качественных выборок
from sklearn.model_selection import train_test_split

# Загрузка данных
data = pd.read_csv('mnist_test.csv', header=None)

# Фильтрация данных для цифр 6, 1, 7
filtered_data = data[data[0].isin([6, 1, 7])]

# Разделение на обучающую и тестовую выборки
train_data = pd.DataFrame()
test_data = pd.DataFrame()

for digit in [6, 1, 7]:
    # Получение всех записей для данной цифры
    digit_data = filtered_data[filtered_data[0] == digit]
    # Разделение на обучающую и тестовую выборку 20 образцов и 10 для проверки
    train_digit, test_digit = train_test_split(digit_data, train_size=20, test_size=10, random_state=42)
    train_data = pd.concat([train_data, train_digit])
    test_data = pd.concat([test_data, test_digit])

# Перемешивание данных
train_data = train_data.sample(frac=1).reset_index(drop=True)
test_data = test_data.sample(frac=1).reset_index(drop=True)

# Сохранение обучающих данных в CSV файл
train_data.to_csv('train_data.csv', index=False, header=False)

# Сохранение тестовых данных в CSV файл
test_data.to_csv('test_data.csv', index=False,header=False)