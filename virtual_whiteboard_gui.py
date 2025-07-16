import sys
import cv2
import numpy as np
import mediapipe as mp
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QWidget, QVBoxLayout, QFileDialog
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

class WhiteboardApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🖐️ Virtual Whiteboard with Hand Gestures")
        self.setGeometry(100, 100, 700, 520)

        # UI Elements
        self.image_label = QLabel()
        self.image_label.setFixedSize(640, 480)

        self.btn_start = QPushButton("▶️ Start")
        self.btn_save = QPushButton("💾 Save Drawing")
        self.btn_stop = QPushButton("⏹️ Stop")

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_save)
        layout.addWidget(self.btn_stop)
        self.setLayout(layout)

        # Connections
        self.btn_start.clicked.connect(self.start_camera)
        self.btn_stop.clicked.connect(self.stop_camera)
        self.btn_save.clicked.connect(self.save_drawing)

        # Timer for real-time capture
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # MediaPipe & Drawing
        self.cap = None
        self.hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.draw_util = mp.solutions.drawing_utils
        self.canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        self.prev_x, self.prev_y = 0, 0

    def fingers_up(self, landmarks):
        tips = [4, 8, 12, 16, 20]
        fingers = []
        # Thumb
        fingers.append(1 if landmarks[tips[0]].x < landmarks[tips[0] - 1].x else 0)
        # Others
        for i in range(1, 5):
            fingers.append(1 if landmarks[tips[i]].y < landmarks[tips[i] - 2].y else 0)
        return fingers

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.start(10)

    def stop_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.image_label.clear()

    def save_drawing(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Drawing", "", "PNG Files (*.png)")
        if file_path:
            cv2.imwrite(file_path, self.canvas)
            print("Drawing saved to:", file_path)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.draw_util.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                fingers = self.fingers_up(hand_landmarks.landmark)

                idx_x = int(hand_landmarks.landmark[8].x * 640)
                idx_y = int(hand_landmarks.landmark[8].y * 480)

                # Drawing mode
                if fingers[1] == 1 and fingers[2] == 0:
                    if self.prev_x == 0 and self.prev_y == 0:
                        self.prev_x, self.prev_y = idx_x, idx_y
                    cv2.line(self.canvas, (self.prev_x, self.prev_y), (idx_x, idx_y), (255, 0, 255), 5)
                    self.prev_x, self.prev_y = idx_x, idx_y
                # Eraser mode
                elif fingers[1] == 1 and fingers[2] == 1:
                    cv2.circle(self.canvas, (idx_x, idx_y), 20, (0, 0, 0), cv2.FILLED)
                    self.prev_x, self.prev_y = 0, 0
                else:
                    self.prev_x, self.prev_y = 0, 0

        # Merge canvas and frame
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, inv = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, inv)
        frame = cv2.bitwise_or(frame, self.canvas)

        # Convert to Qt format and display
        img = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        self.image_label.setPixmap(pix)

    def closeEvent(self, event):
        self.stop_camera()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = WhiteboardApp()
    win.show()
    sys.exit(app.exec_())
