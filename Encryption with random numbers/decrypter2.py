import json
import random
import math
import spacy

nlp = spacy.load('en_core_web_sm')

# Функция для проверки наличия слов или предложений в тексте
def check_text(text):
    # Разбить текст на предложения
    doc = nlp(text)
    sentences = list(doc.sents)

    # Если нет предложений, то и слов тоже нет
    if not sentences:
        return False

    # Проверить каждое предложение на наличие слов
    for sentence in sentences:
        words = [token.text for token in sentence if not token.is_punct and not token.is_space]
        # Если есть хотя бы одно слово, то в тексте есть слова и предложения
        if words:
            return True

    # Если все предложения были пустыми, то в тексте нет слов и предложений
    return False

def has_english_letters(lst):
    for word in lst:
        for char in word:
            if char.isalpha() and char.isascii():
                return True
    return False

# Читаем список из файла json
with open("encrypted_numbers.json", "r") as f:
    encrypted_numbers = json.load(f)

# Извлекаем корень из каждого элемента списка
square_root_numbers = [math.sqrt(n) for n in encrypted_numbers]

# Сохраняем копию списка
original_numbers = square_root_numbers.copy()

# Генерируем случайный коэффициент
random_coefficient = random.randint(1, 10)

# Делим каждый элемент списка на случайный коэффициент
divided_numbers = [n / random_coefficient for n in square_root_numbers]

# Извлекаем корень из каждого элемента списка
decrypted_numbers = [math.sqrt(n) for n in divided_numbers]

# Объединяем числа в текст
decrypted_text = "".join([chr(round(n)) for n in decrypted_numbers])

# Проверяем текст на наличие слов или предложений
while not check_text(decrypted_text):
    # Возвращаем список к сохраненному состоянию
    square_root_numbers = original_numbers.copy()

    # Генерируем новый случайный коэффициент
    random_coefficient = random.randint(1, 10)

    # Делим каждый элемент списка на новый случайный коэффициент
    divided_numbers = [n / random_coefficient for n in square_root_numbers]

    # Извлекаем корень из каждого элемента списка
    decrypted_numbers = [math.sqrt(n) for n in divided_numbers]

    # Объединяем числа в текст
    decrypted_text = "".join([chr(round(n)) for n in decrypted_numbers])

# Выводим расшифрованный текст
print("Расшифрованный текст:", decrypted_text)
