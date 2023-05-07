import json
import random

# Вводим текст для шифрования
text = input("Введите текст для шифрования: ")

# Создаем список, в котором будут храниться числа
numbers = []

# Проходим по каждому символу текста
for char in text:
    # Если символ - число, то добавляем его в список
    if char.isdigit():
        numbers.append(int(char))
    # Если символ не число, то находим его номер в таблице ASCII и добавляем в список
    else:
        numbers.append(ord(char))

# Возведем каждый элемент списка в квадрат
squared_numbers = [n**2 for n in numbers]

# Генерируем случайный коэффициент
random_coefficient = random.randint(1, 10)

# Умножим каждый элемент списка на случайный коэффициент
multiplied_numbers = [n * random_coefficient for n in squared_numbers]

# Возведем каждый элемент списка в квадрат еще раз
encrypted_numbers = [n**2 for n in multiplied_numbers]

# Сохраняем список в файле формата json
with open("encrypted_numbers.json", "w") as f:
    json.dump(encrypted_numbers, f)
