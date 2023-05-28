import cv2
import face_recognition
import pickle
import numpy as np

def recognize_faces():
    known_faces = {}

    # Загрузка обученных данных для каждого человека
    while True:
        name = input("Введите имя человека (или 'q' для выхода): ")
        if name == 'q':
            break
        
        file_name = f"{name}_data.pkl"
        try:
            with open(file_name, 'rb') as file:
                face_encodings = pickle.load(file)
            known_faces[name] = face_encodings
            print(f"Обученные данные для '{name}' успешно загружены.")
        except FileNotFoundError:
            print(f"Файл с данными для '{name}' не найден.")

    video_capture = cv2.VideoCapture(0)  # Используется индекс 0 для основной камеры

    while True:
        ret, frame = video_capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        used_names = {}  # Словарь использованных имен

        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Поиск совпадений среди обученных лиц
            matches = face_recognition.compare_faces(list(known_faces.values()), face_encoding)
            matched_indexes = np.where(matches)[0]
            names = []

            # Если найдено совпадение, выбираем имена
            if len(matched_indexes) > 0:
                names = [list(known_faces.keys())[index] for index in matched_indexes]

            available_names = [name for name in names if name not in used_names.values()]  # Выбираем только доступные имена

            if len(available_names) > 0:
                name = available_names[0]  # Берем первое доступное имя
                face_id = id(face_location)  # Генерируем уникальный идентификатор для лица

                if face_id in used_names:
                    used_names[face_id] = name  # Обновляем имя для существующего лица
                else:
                    used_names[face_id] = name  # Добавляем новое имя для нового лица
            else:
                name = "Unknown"  # Если нет доступных имен, выводим "Unknown" или другое сообщение

            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Удаление имен, связанных с невидимыми лицами
            invisible_faces = set(used_names.keys()) - set(id(face_location) for face_location in face_locations)
            for face_id in invisible_faces:
                del used_names[face_id]

        cv2.imshow('Face Recognition', frame)
            
        # Выход из цикла по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    recognize_faces()
