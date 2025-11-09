import sys
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QFileDialog, QHBoxLayout, QMessageBox, QComboBox
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import QTimer, Qt
from ultralytics import YOLO

# Import Light Theme stylesheet
from style_light import light_theme


class VehicleDetector(QWidget):
    def __init__(self):
        super().__init__()

        # Window Setup
        self.setWindowTitle("Vehicle Detector - Built by Prabhant Tiwari")
        self.setGeometry(200, 100, 950, 650)

        # Apply the light theme styling from external file
        self.setStyleSheet(light_theme)

        # Load YOLO model
        self.model = YOLO("yolov8n.pt")

        # UI COMPONENTS 

        #Header Title
        self.header = QLabel("🚗 Vehicle Detector")
        self.header.setAlignment(Qt.AlignCenter)
        self.header.setFont(QFont("Segoe UI", 20, QFont.Bold))
        self.header.setFixedHeight(70)
        self.header.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                            stop:0 #74b9ff, stop:1 #a29bfe);
                color: white;
                border-radius: 10px;
                font-weight: bold;
                letter-spacing: 1px;
            }
        """)

        #  Video Display Area
        self.video_label = QLabel("Live feed will appear here.")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(880, 480)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #ffffff;
                border: 2px solid #dee2e6;
                border-radius: 10px;
            }
        """)

        # Source Selector and Buttons
        self.source_selector = QComboBox()
        self.source_selector.addItems(["Webcam", "Upload Video File"])

        self.start_button = QPushButton("▶ Start Detection")
        self.stop_button = QPushButton("⏹ Stop Detection")
        self.stop_button.setEnabled(False)

        # Horizontal Layout for top control bar
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.source_selector)
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)

        # --- Footer Credit ---
        self.credit_label = QLabel("✨ Built by Prabhant Tiwari")
        self.credit_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.credit_label.setAlignment(Qt.AlignCenter)
        self.credit_label.setStyleSheet("color: #6c757d; margin-top: 8px;")

        # --- Main Layout ---
        layout = QVBoxLayout()
        layout.addWidget(self.header)
        layout.addLayout(control_layout)
        layout.addWidget(self.video_label)
        layout.addWidget(self.credit_label)
        self.setLayout(layout)

        # Timer for video frame update
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.cap = None

        # Button actions
        self.start_button.clicked.connect(self.start_detection)
        self.stop_button.clicked.connect(self.stop_detection)

    # ---------------- MAIN LOGIC from here 

    def start_detection(self):
        """Start detection from either webcam or uploaded video file."""
        source_choice = self.source_selector.currentText()

        if source_choice == "Webcam":
            self.cap = cv2.VideoCapture(0)
        else:
            # Open file chooser dialog for video upload
            video_path, _ = QFileDialog.getOpenFileName(
                self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)"
            )
            if not video_path:
                QMessageBox.warning(self, "No File Selected", "Please select a valid video file.")
                return
            self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Unable to open video source!")
            return

        # Enable / Disable appropriate buttons
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.timer.start(30)  # Timer controls frame refresh rate

    def stop_detection(self):
        """Stop the video feed and detection."""
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.video_label.setText("Detection stopped.")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def update_frame(self):
        """Read each frame, perform detection, and update on UI."""
        ret, frame = self.cap.read()
        if not ret:
            self.stop_detection()
            return

        # Resize frame for smoother performance
        frame = cv2.resize(frame, (800, 480))

        # Perform YOLOv8 Detection
        results = self.model(frame, conf=0.5, verbose=False)

        car_count, bike_count = 0, 0
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 2:  # Car
                    car_count += 1
                    color, label = (0, 255, 0), "Car"
                elif cls == 3:  # Bike
                    bike_count += 1
                    color, label = (255, 0, 0), "Bike"
                else:
                    continue

                # Draw rectangle and label on detected objects
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Display Count Text on frame
        cv2.putText(frame, f"Cars: {car_count} | Bikes: {bike_count}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 50, 50), 3)

        # Convert frame for PyQt display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img))


# MAIN EXECUTION 
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("Vehicle Detector - Prabhant Tiwari")
    window = VehicleDetector()
    window.show()
    sys.exit(app.exec_())
