import cv2
import face_recognition
import pickle

def train_faces():
    known_faces = {}
    video_capture = cv2.VideoCapture(0)  # Используется индекс 0 для основной камеры
    
    while True:
        name = input("Введите имя для этого человека (или 'q' для выхода): ")
        if name == 'q':
            break
        
        face_encodings = []

        print(f"Пожалуйста, поверните лицо в разных позах для обучения '{name}' (например, вправо, влево, вверх, вниз). Запись будет продолжаться в течение 30 секунд.")

        while True:
            ret, frame = video_capture.read()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)

            if len(face_locations) > 0:
                for (top, right, bottom, left) in face_locations:
                    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
                
                cv2.imshow('Face Training', frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('s'):
                    face_encodings.append(face_recognition.face_encodings(rgb_frame, face_locations)[0])
                    print(f"Записано {len(face_encodings)} изображений лица для '{name}'.")
                    
                if len(face_encodings) >= 30:  # Записываем 30 изображений лица для каждого человека
                    break
            
        known_faces[name] = face_encodings

        file_name = f"{name}_data.pkl"
        with open(file_name, 'wb') as file:
            pickle.dump(face_encodings, file)
        print(f"Данные для '{name}' сохранены в файл {file_name}.")

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    train_faces()
