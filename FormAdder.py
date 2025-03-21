from PySide6.QtWidgets import QApplication, QDialog, QPushButton, QVBoxLayout, QTableWidget, QTableWidgetItem
from Adder import AdderDialog

class FormAdder(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Form Adder")
        self.setGeometry(100, 100, 400, 300)

        # Tạo bảng hiển thị dữ liệu
        self.table = QTableWidget(self)
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Label", "Source Direction"])
        self.table.setRowCount(0)

        # Các nút bấm
        self.btn_add = QPushButton("Add", self)
        self.btn_save = QPushButton("Save", self)
        self.btn_cancel = QPushButton("Cancel", self)

        # Bố cục
        layout = QVBoxLayout(self)
        layout.addWidget(self.table)
        layout.addWidget(self.btn_add)
        layout.addWidget(self.btn_save)
        layout.addWidget(self.btn_cancel)

        # Kết nối sự kiện
        self.btn_add.clicked.connect(self.open_adder)
        self.btn_save.clicked.connect(self.save_data)
        self.btn_cancel.clicked.connect(self.close)

    def open_adder(self):
        dialog = AdderDialog(self)
        if dialog.exec():
            label, source = dialog.get_data()
            if label and source:
                row = self.table.rowCount()
                self.table.insertRow(row)
                self.table.setItem(row, 0, QTableWidgetItem(label))
                self.table.setItem(row, 1, QTableWidgetItem(source))

    def data_to_dict(self):
        data = {}
        for row in range(self.table.rowCount()):
            label = self.table.item(row, 0).text()
            source = self.table.item(row, 1).text()
            data[label] = source
        return data
    def save_data(self):
        data = self.data_to_dict()
        print(data)
        # Lưu dữ liệu to labels.txt in the same folder
        with open("labels.txt", "w") as file:
            for label, source in data.items():
                file.write(f"{label} {source}\n")
        self.close()
        
if __name__ == "__main__":
    app = QApplication([])
    window = FormAdder()
    window.show()
    app.exec()
