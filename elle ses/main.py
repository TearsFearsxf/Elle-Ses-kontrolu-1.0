import cv2
import mediapipe as mp
from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume
import tkinter as tk

class HandGestureApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.current_volume = 0.5
        self.command_label = tk.Label(root, text="Command: None")
        self.command_label.pack()

    def update_volume(self, hand_landmarks):
        if hand_landmarks:
            thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]

            thumb_tip_x, thumb_tip_y = thumb_tip.x, thumb_tip.y
            index_tip_x, index_tip_y = index_tip.x, index_tip.y

            # İki parmak arasındaki uzaklığı hesapla
            finger_distance = ((thumb_tip_x - index_tip_x) ** 2 + (thumb_tip_y - index_tip_y) ** 2) ** 0.5

            # Ses seviyesini güncelleme
            volume = 0.1 + 0.9 * (finger_distance - 0.05) / 0.15
            volume = max(0, min(1, volume))
            self.current_volume = volume

            sessions = AudioUtilities.GetAllSessions()
            for session in sessions:
                volume_interface = session._ctl.QueryInterface(ISimpleAudioVolume)
                volume_interface.SetMasterVolume(self.current_volume, None)

    def show_command_indicator(self, command):
        # Yapılan komutu ekranda gösteren bir gösterge ekleyin
        self.command_label.config(text="Command: {}".format(command))

    def recognize_command(self, hand_landmarks):
        # Örnek bir komut tanıma
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]

        # El parmakları arasındaki uzaklığı hesapla
        finger_distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5

        if finger_distance < 0.05:
            return "Volume Up"
        elif finger_distance > 0.2:
            return "Volume Down"
        elif 0.02 < finger_distance < 0.05:  # İki işaret parmağı birbirine yaklaştığında
            return "Pause"
        else:
            return "No Command"

    def main_loop(self):
        while self.cap.isOpened():
            ret, image = self.cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # Ses seviyesini güncelle
                self.update_volume(hand_landmarks)

                # Komut tanıma
                command = self.recognize_command(hand_landmarks)

                # Komutu ekranda göster
                self.show_command_indicator(command)

                # Tanılanan komuta göre işlem yap
                if command == "Pause":
                    print("Video Paused")

            cv2.imshow('Hand Gesture Recognition', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = HandGestureApp()
    root.after(100, app.main_loop)
    root.mainloop()
