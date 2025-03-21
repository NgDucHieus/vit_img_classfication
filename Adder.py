from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QHBoxLayout

class AdderDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Adder")
        self.setGeometry(200, 200, 300, 200)

        # Các thành phần UI
        self.label_label = QLabel("Tên danh mục:")
        self.input_label = QLineEdit(self)

        self.label_source = QLabel("Data mẫu:")
        self.input_source = QLineEdit(self)

        self.btn_ok = QPushButton("OK")
        self.btn_cancel = QPushButton("Cancel")

        # Bố cục
        layout = QVBoxLayout(self)
        layout.addWidget(self.label_label)
        layout.addWidget(self.input_label)
        layout.addWidget(self.label_source)
        layout.addWidget(self.input_source)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_ok)
        btn_layout.addWidget(self.btn_cancel)

        layout.addLayout(btn_layout)

        # Kết nối sự kiện
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

    def get_data(self):
        return self.input_label.text(), self.input_source.text()
