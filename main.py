import sys
import os
import shutil
import platform
import subprocess
import json
import logging
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QLineEdit, QFileDialog, QProgressBar, QLabel,
    QFrame, QHBoxLayout, QMessageBox
)
from typing import Optional, List
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QIcon
from PySide6.QtCore import QSize
import easyocr
import cv2
import re  # Để xử lý tên file

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="image_classification.log",
)

class ImageClassificationWorker(QThread):
    progress = Signal(int)
    status = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, source_folder, destination_folder):
        super().__init__()
        self.source_folder = source_folder
        self.destination_folder = destination_folder
        self.stopped = False
        # Khởi tạo EasyOCR ngay trong __init__
        self.reader = easyocr.Reader(['en', 'vi'], gpu=False)

    def run(self):
        try:
            self.status.emit("Initializing...")
            self.progress.emit(5)

            # Import heavy modules inside the thread
            import torch
            import numpy as np
            from PIL import Image

            try:
                from transformers import AutoProcessor, AutoModel
                import torch.nn.functional as F
            except ImportError:
                self.error.emit(
                    "Required packages not installed. Please install transformers.")
                return

            self.status.emit("Loading AI model...")
            self.progress.emit(10)
            dir_path = os.path.dirname(os.path.realpath(__file__))
            model_path = os.path.join(dir_path, "model")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logging.info(f"Using device: {device}")
            try:
                processor = AutoProcessor.from_pretrained(model_path)
                model = AutoModel.from_pretrained(model_path).to(device)
                model.eval()
                self.progress.emit(30)
            except Exception as e:
                self.error.emit(f"Failed to load the AI model: {str(e)}")
                return

            try:
                with open("labels.json", "r") as f:
                    category_folders = json.load(f)
            except FileNotFoundError:
                category_folders = [
                    "1.Metal sheet Roof",
                    "2.Overview",
                    "3.Structure Connection",
                    "4.Tape & Caliper Dimension",
                    "5.Thickness Tester",
                    "6.Hardness Tester",
                    "7.Concrete Roof"
                ]
                logging.warning("labels.json not found, using default category folders.")

            categories = category_folders
            confidence_threshold = 0.3
            example_images_folder = os.path.join(dir_path, "data_sample")

            self.status.emit("Creating output folders...")
            self.progress.emit(35)

            for category in categories + ["Skipped"]:
                label_folder = os.path.join(self.destination_folder, category)
                os.makedirs(label_folder, exist_ok=True)

            def extract_clip_features(image_path: str, processor, model) -> Optional[torch.Tensor]:
                try:
                    image = Image.open(image_path)
                    inputs = processor(images=image, return_tensors="pt", padding=True).to(device)
                    with torch.no_grad():
                        features = model.get_image_features(**inputs)
                        features = F.normalize(features, p=2, dim=-1)
                    return features.squeeze()
                except Exception as e:
                    logging.error(f"Error processing {image_path}: {e}")
                    return None

            def load_example_image_features(example_images_folder, categories, processor, model) -> dict:
                category_prototype_features = {}
                for category_folder_name in categories:
                    category_path = os.path.join(example_images_folder, category_folder_name)
                    if not os.path.isdir(category_path):
                        logging.warning(f"Category folder '{category_path}' not found.")
                        continue

                    example_image_paths = [
                        os.path.join(category_path, f) for f in os.listdir(category_path)
                        if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))
                    ]
                    if not example_image_paths:
                        logging.warning(f"No example images found in '{category_path}'.")
                        continue

                    example_features_list = []
                    for image_path in example_image_paths:
                        features = extract_clip_features(image_path, processor, model)
                        if features is not None:
                            example_features_list.append(features)

                    if example_features_list:
                        category_features = torch.stack(example_features_list).mean(dim=0)
                        category_prototype_features[category_folder_name] = F.normalize(category_features, p=2, dim=-1).cpu()
                    else:
                        logging.warning(f"No features extracted for example images in '{category_path}'.")

                return category_prototype_features

            category_prototype_embeddings = load_example_image_features(example_images_folder, categories, processor, model)
            logging.info(f"Loaded prototype features for categories: {list(category_prototype_embeddings.keys())}")

            def extract_batch_features(image_paths: List[str], processor, model) -> Optional[torch.Tensor]:
                images, valid_paths = [], []
                for path in image_paths:
                    try:
                        image = Image.open(path)
                        images.append(image)
                        valid_paths.append(path)
                    except Exception as e:
                        logging.warning(f"Skipping {path}: {e}")
                if not images:
                    return None
                inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    features = model.get_image_features(**inputs)
                    features = F.normalize(features, p=2, dim=-1)
                return features if len(features) > 0 else None

            def extract_text(image_path: str) -> str:
                try:
                    # Đọc ảnh bằng OpenCV
                    img = cv2.imread(image_path)
                    if img is None:
                        raise ValueError("Không thể đọc ảnh.")

                    # Lấy kích thước ảnh
                    height, width = img.shape[:2]

                    # Xác định tọa độ vùng cắt (ROI) rộng hơn
                    x_start = int(width * 0.6)   # 60% chiều rộng từ trái qua
                    x_end = int(width * 1.0)     # 100% chiều rộng (đến mép phải)
                    y_start = int(height * 0.70) # 70% chiều cao từ trên xuống
                    y_end = int(height * 1.0)    # 100% chiều cao, sát mép dưới

                    # Cắt vùng ROI
                    roi = img[y_start:y_end, x_start:x_end]

                    # Chuyển sang ảnh xám
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                    # Chuyển lại thành định dạng PIL để EasyOCR xử lý
                    image = Image.fromarray(gray)

                    # Sử dụng EasyOCR để nhận diện văn bản từ vùng ROI
                    allowlist = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ°,. Hà Nội Vĩnh Phúc KATOLEC HERA Xưởng Dệt TCE VINA DENIM PKN Đại Lộc Nhà ăn Ăn CẨM NGUYÊN Tiên Phước Tổng quan Nghệ An Xưởng Cán MỘC BÀI SUNJIN Nhà máy Sợi An Ninh XƯỞNG CÁN Nhuộm Phụ"
                    results = self.reader.readtext(
                        np.array(image),
                        allowlist=allowlist,
                        min_size=10,
                        text_threshold=0.7,
                        low_text=0.3,
                        paragraph=False  # Không nhóm thành đoạn, nhận diện từng dòng riêng lẻ
                    )

                    if not results:
                        return "No text extracted."

                    # Sắp xếp các đoạn văn bản theo tọa độ y (từ trên xuống dưới)
                    sorted_results = sorted(results, key=lambda x: x[0][0][1])

                    # Tách các dòng dựa trên khoảng cách tọa độ y
                    lines = []
                    current_line = []
                    y_threshold = 10  # Ngưỡng khoảng cách y để tách dòng

                    for i, result in enumerate(sorted_results):
                        y_coord = result[0][0][1]
                        x_coord = result[0][0][0]
                        text = result[1]

                        if i == 0:
                            current_line.append((x_coord, text))
                        else:
                            prev_y_coord = sorted_results[i-1][0][0][1]
                            if y_coord - prev_y_coord > y_threshold:
                                # Sắp xếp các đoạn trong dòng hiện tại theo tọa độ x
                                current_line.sort(key=lambda x: x[0])
                                lines.append(" ".join([item[1] for item in current_line]))
                                current_line = [(x_coord, text)]
                            else:
                                current_line.append((x_coord, text))

                    # Thêm dòng cuối cùng
                    if current_line:
                        current_line.sort(key=lambda x: x[0])
                        lines.append(" ".join([item[1] for item in current_line]))

                    # Lấy hai dòng cuối
                    last_two_lines = lines[-2:] if len(lines) >= 2 else lines
                    last_two_text = " ".join(last_two_lines)
                    return last_two_text.strip()

                except Exception as e:
                    logging.error(f"Error extracting text from {image_path}: {e}")
                    return f"Error extracting text: {str(e)}"

            self.status.emit("Finding images...")
            image_files = [
                os.path.join(self.source_folder, f) for f in os.listdir(self.source_folder)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))
            ]

            if not image_files:
                self.status.emit("No images found!")
                self.error.emit("No images found in the source folder.")
                return

            total_images = len(image_files)
            self.status.emit(f"Processing {total_images} images...")
            batch_size = 10

            for i in range(0, len(image_files), batch_size):
                if self.stopped:
                    return

                batch_paths = image_files[i:i + batch_size]
                batch_features = extract_batch_features(batch_paths, processor, model)
                if batch_features is None:
                    continue

                for path, image_feature in zip(batch_paths, batch_features):
                    filename = os.path.basename(path)
                    self.status.emit(f"Processing {filename} ({i+1}/{total_images})")

                    max_similarity = -1.0
                    predicted_category = "Skipped"
                    confidence_value = 0.0

                    for category_name, prototype_feature in category_prototype_embeddings.items():
                        similarity_score = F.cosine_similarity(image_feature.cpu(), prototype_feature.unsqueeze(0)).item()
                        if similarity_score > max_similarity:
                            max_similarity = similarity_score
                            predicted_category = category_name
                            confidence_value = similarity_score

                    destination_folder = predicted_category if confidence_value >= confidence_threshold else "Skipped"
                    destination_path = os.path.join(self.destination_folder, destination_folder)

                    # Trích xuất văn bản từ ảnh (hai dòng cuối)
                    text = extract_text(path)
                    print(f"\nImage: {filename}")
                    print(f"Category: {predicted_category} (confidence: {confidence_value:.4f})")
                    print(f"Extracted Text (last 2 lines): {text}")

                    # Xác định tên file mới dựa trên văn bản OCR
                    original_ext = os.path.splitext(filename)[1]  # Lấy phần mở rộng (.jpg, .png, v.v.)
                    if text and text != "No text extracted." and not text.startswith("Error extracting text:"):
                        # Loại bỏ ký tự không hợp lệ trong tên file
                        new_filename = re.sub(r'[<>:"/\\|?*]', '', text)[:100] + original_ext  # Giới hạn 100 ký tự
                    else:
                        new_filename = filename  # Giữ nguyên tên gốc nếu không có văn bản hợp lệ

                    # Đường dẫn đích với tên file mới
                    destination_file_path = os.path.join(destination_path, new_filename)

                    try:
                        shutil.copy(path, destination_file_path)
                        log_msg = (
                            f"Classified {filename} as '{predicted_category}' (confidence: {confidence_value:.4f}), copied to {destination_file_path}"
                            if destination_folder != "Skipped"
                            else f"Skipped {filename}: low confidence ({confidence_value:.4f}) for '{predicted_category}', copied to {destination_file_path}"
                        )
                        logging.info(log_msg)
                    except Exception as e:
                        logging.error(f"Error processing {filename}: {str(e)}")

                progress_value = 40 + int((i + batch_size) / total_images * 55) if (i + batch_size) <= total_images else 95
                self.progress.emit(min(progress_value, 99))

            self.status.emit("Classification complete!")
            self.progress.emit(100)
            self.finished.emit()

        except Exception as e:
            self.error.emit(f"An unexpected error occurred: {str(e)}")
            logging.exception("Unexpected error in worker thread")

    def stop(self):
        self.stopped = True

def open_folder(folder_path: str):
    try:
        if platform.system() == "Windows":
            os.startfile(folder_path)
        elif platform.system() == "Darwin":
            subprocess.run(["open", folder_path])
        elif platform.system() == "Linux":
            subprocess.run(["xdg-open", folder_path])
        else:
            logging.warning(f"Unsupported OS: {platform.system()}. Cannot open folder.")
            return
        logging.info(f"Opened folder: {folder_path}")
    except Exception as e:
        logging.error(f"Failed to open folder {folder_path}: {e}")

class RoundedFrame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("roundedFrame")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width(), self.height(), 12, 12)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#ffffff"))
        painter.drawPath(path)

class ImageClassificationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.worker = None

    def initUI(self):
        self.setWindowTitle("Image Classification Tool")
        app_icon_path = "app_icon.png"
        if os.path.exists(app_icon_path):
            app_icon = QIcon(app_icon_path)
            self.setWindowIcon(app_icon)
        else:
            logging.warning(f"Icon file not found at: {app_icon_path}")
        self.setFixedSize(600, 450)

        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 #f5f7fa, stop:1 #e9eef6);
                font-family: 'Segoe UI', Arial, sans-serif;
            }
        """)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(25, 25, 25, 25)
        main_layout.setSpacing(20)

        container = RoundedFrame()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(25, 25, 25, 25)
        container_layout.setSpacing(20)

        title_label = QLabel("Image Classification Tool")
        title_label.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #333333;")
        container_layout.addWidget(title_label)

        source_label = QLabel("Source Folder:")
        source_label.setStyleSheet("font-weight: bold; color: #555555;")
        container_layout.addWidget(source_label)

        source_frame = QFrame()
        source_frame.setStyleSheet("""
            QFrame {
                border: 1px solid #dcdcdc;
                border-radius: 8px;
                background-color: #ffffff;
            }
        """)
        source_layout = QHBoxLayout(source_frame)
        source_layout.setContentsMargins(12, 12, 12, 12)

        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("Select source folder...")
        self.source_edit.setReadOnly(True)
        self.source_edit.setMinimumHeight(14)
        self.source_edit.setStyleSheet("border: none; font-size: 14px; background: transparent; color: black;")
        source_browse = QPushButton()
        source_browse.setIcon(QIcon.fromTheme("folder"))
        source_browse.setIconSize(QSize(18, 18))
        source_browse.setFixedSize(30, 30)
        source_browse.setStyleSheet("background-color: transparent; border: none; border-radius:5px; margin-left: 10px; margin-bottom: 15px")
        source_browse.clicked.connect(self.select_source_folder)

        source_layout.addWidget(self.source_edit)
        source_layout.addWidget(source_browse)
        container_layout.addWidget(source_frame)

        dest_label = QLabel("Destination Folder:")
        dest_label.setStyleSheet("font-weight: bold; color: #555555;")
        container_layout.addWidget(dest_label)

        dest_frame = QFrame()
        dest_frame.setStyleSheet("""
            QFrame {
                border: 1px solid #dcdcdc;
                border-radius: 8px;
                background-color: #ffffff;
            }
        """)
        dest_layout = QHBoxLayout(dest_frame)
        dest_layout.setContentsMargins(12, 12, 12, 12)

        self.dest_edit = QLineEdit()
        self.dest_edit.setPlaceholderText("Select destination folder...")
        self.dest_edit.setReadOnly(True)
        self.dest_edit.setMinimumHeight(14)
        self.dest_edit.setStyleSheet("border: none; font-size: 14px; background: transparent; color: black;")
        dest_browse = QPushButton()
        dest_browse.setIcon(QIcon.fromTheme("folder"))
        dest_browse.setIconSize(QSize(18, 18))
        dest_browse.setFixedSize(30, 30)
        dest_browse.setStyleSheet("background-color: transparent; border: none; border-radius:5px; margin-left: 10px; margin-bottom: 15px")
        dest_browse.clicked.connect(self.select_dest_folder)

        dest_layout.addWidget(self.dest_edit)
        dest_layout.addWidget(dest_browse)
        container_layout.addWidget(dest_frame)

        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #555555; font-size: 14px;")
        container_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 5px;
                background-color: #e0e0e0;
                height: 10px;
                text-align: center;
            }
            QProgressBar::chunk {
                border-radius: 5px;
                background-color: #4a86e8;
            }
        """)
        container_layout.addWidget(self.progress_bar)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)

        self.start_button = QPushButton("Start Classification")
        self.start_button.setEnabled(False)
        self.start_button.setStyleSheet("background-color: #5cb85c; color: white; padding: 10px; border-radius: 8px; font-weight: bold;")
        self.start_button.clicked.connect(self.start_classification)

        self.open_button = QPushButton("Open Results")
        self.open_button.setEnabled(False)
        self.open_button.setStyleSheet("background-color: #f0ad4e; color: white; padding: 10px; border-radius: 8px; font-weight: bold;")
        self.open_button.clicked.connect(self.open_results)

        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.open_button)
        container_layout.addLayout(button_layout)

        main_layout.addWidget(container)
        self.setLayout(main_layout)

    def select_source_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Source Folder")
        if folder:
            self.source_edit.setText(folder)
            self.check_ready()

    def select_dest_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Destination Folder")
        if folder:
            self.dest_edit.setText(folder)
            self.check_ready()

    def check_ready(self):
        if self.source_edit.text() and self.dest_edit.text():
            self.start_button.setEnabled(True)
        else:
            self.start_button.setEnabled(False)

    def start_classification(self):
        source_folder = self.source_edit.text()
        dest_folder = self.dest_edit.text()

        if not os.path.exists(source_folder):
            QMessageBox.warning(self, "Error", "Source folder does not exist.")
            return

        has_images = False
        for file in os.listdir(source_folder):
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                has_images = True
                break

        if not has_images:
            QMessageBox.warning(self, "No Images Found", "No image files found in the source folder.")
            return

        self.start_button.setEnabled(False)
        self.source_edit.setEnabled(False)
        self.dest_edit.setEnabled(False)

        self.worker = ImageClassificationWorker(source_folder, dest_folder)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(self.status_label.setText)
        self.worker.finished.connect(self.on_classification_finished)
        self.worker.error.connect(self.on_classification_error)
        self.worker.start()

    def on_classification_finished(self):
        self.source_edit.setEnabled(True)
        self.dest_edit.setEnabled(True)
        self.start_button.setEnabled(True)
        self.open_button.setEnabled(True)
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle("Success")
        msg_box.setText("Phân loại ảnh thành công!")
        msg_box.setStyleSheet("""
            QMessageBox { background-color: white; }
            QLabel { color: black; font-size: 14px; }
            QPushButton { background-color: #4CAF50; color: white; padding: 5px; border-radius: 5px; }
        """)
        msg_box.exec()

    def on_classification_error(self, error_message):
        QMessageBox.critical(self, "Error", error_message)
        self.source_edit.setEnabled(True)
        self.dest_edit.setEnabled(True)
        self.start_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready")

    def open_results(self):
        dest_folder = self.dest_edit.text()
        if os.path.exists(dest_folder):
            open_folder(dest_folder)
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Thư mục lưu trữ không tồn tại!")
            msg.setWindowTitle("Error")
            msg.setStyleSheet("QLabel { color: black; }")
            msg.exec()

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app_icon_path = "icon.png"
    if os.path.exists(app_icon_path):
        app_icon = QIcon(app_icon_path)
        app.setWindowIcon(app_icon)
    window = ImageClassificationApp()
    window.show()
    sys.exit(app.exec())