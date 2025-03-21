import os
from PySide6.QtWidgets import QApplication, QDialog, QPushButton, QVBoxLayout, QTableWidget, QTableWidgetItem, QMessageBox
from Adder import AdderDialog

class FormAdder(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Form Adder")
        self.setGeometry(100, 100, 400, 300)

        # Create table to display data
        self.table = QTableWidget(self)
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Label", "Source Direction"])
        self.table.setRowCount(0)

        # Data labels
        self.data = {}
        self.load_data_from_labels_txt() # Load data from labels.txt in __init__

        # Buttons
        self.btn_add = QPushButton("Add", self)
        self.btn_save = QPushButton("Save", self)
        self.btn_cancel = QPushButton("Cancel", self)

        # Layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.table)
        layout.addWidget(self.btn_add)
        layout.addWidget(self.btn_save)
        layout.addWidget(self.btn_cancel)

        # Event connections
        self.btn_add.clicked.connect(self.open_adder)
        self.btn_save.clicked.connect(lambda: self.save_data_to_labels_txt('save'))  # Corrected connection for Save
        self.btn_cancel.clicked.connect(lambda: self.save_data_to_labels_txt('cancel')) # Corrected connection for Cancel

    def load_data_from_labels_txt(self):
        """Loads data from labels.txt and populates the table and self.data."""
        if os.path.exists("labels.txt") and os.path.getsize("labels.txt") > 0:
            try:
                with open("labels.txt", "r", encoding="utf-8") as file: # Explicitly specify encoding
                    print('get by labels.txt')
                    self.table.setRowCount(0) # Clear existing rows before loading
                    self.data = {} # Clear existing data

                    for line_number, line in enumerate(file, 1): # Enumerate for line numbers in error messages
                        line = line.strip()
                        if not line: # Skip empty lines
                            continue

                        parts = line.split(' ', 1) # Split at the first space, max 2 parts
                        if len(parts) == 2:
                            label, source = parts
                            row = self.table.rowCount()
                            self.table.insertRow(row)
                            self.table.setItem(row, 0, QTableWidgetItem(label))
                            self.table.setItem(row, 1, QTableWidgetItem(source))
                            self.data[label] = source
                        else:
                            QMessageBox.warning(self, "Warning",
                                                f"Invalid line format in labels.txt at line {line_number}: '{line}'.\n"
                                                f"Expected 'Label Source', skipping this line.")
                            print(f"Warning: Invalid line format in labels.txt at line {line_number}: '{line}'. Skipping.")
                    self.table.resizeColumnsToContents()
                print("Data loaded from labels.txt successfully.")
                print("Loaded data:", self.data) # Debug print loaded data
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading data from labels.txt: {e}")
                print(f"Error loading data from labels.txt: {e}")
        else:
            print("labels.txt not found or empty, loading from data_sample (if available).")
            self.load_data_from_data_sample() # Fallback to data_sample if labels.txt is missing/empty


    def load_data_from_data_sample(self):
        """Loads data from subdirectories in 'data_sample' if labels.txt is not available."""
        # Debug prints - keep these for troubleshooting
        print("labels.txt not found, trying to read from data_sample")
        print(f"Current working directory: {os.getcwd()}")
        print(f"data_sample exists: {os.path.exists('data_sample')}")

        # If labels.txt doesn't exist, get directories from data_sample
        subdirs = self.get_subdirectory("data_sample")
        print(f"Found subdirectories: {subdirs}")

        if not subdirs:
            print("No subdirectories found in data_sample")
            return # Exit if no subdirectories found

        self.table.setRowCount(0) # Clear existing rows before loading
        self.data = {} # Clear existing data

        for subdir_path, subdir_name in subdirs:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(subdir_name))
            self.table.setItem(row, 1, QTableWidgetItem(subdir_path))
            self.data[subdir_name] = subdir_path
            print(f"Added subdirectory: {subdir_name} -> {subdir_path}")
        self.table.resizeColumnsToContents()
        print("Data loaded from data_sample subdirectories.")
        print("Loaded data:", self.data) # Debug print loaded data


    def data_to_dict(self):
        """Converts table data to a dictionary."""
        data = {}
        for row in range(self.table.rowCount()):
            label = self.table.item(row, 0).text()
            source = self.table.item(row, 1).text()
            data[label] = source
        return data

    def save_data_to_labels_txt(self, type_cmd='save'):
        """
        Saves data from the table to labels.txt.
        Args:
            type_cmd (str): Command type - 'save' or 'cancel'
        """
        data_to_save = self.data_to_dict()
        try:
            with open("labels.txt", "w", encoding="utf-8") as file:
                for label, source in data_to_save.items():
                    file.write(f"{label} {source}\n")

            # Show success message and handle closing based on type_cmd
            if type_cmd == 'save':
                QMessageBox.information(self, "Success", "Data saved successfully!")
                print("Data saved successfully:", data_to_save)
            elif type_cmd == 'cancel':
                print("Data saved before closing (Cancel):", data_to_save) # More accurate message
                self.close() # Close the dialog when Cancel is clicked successfully

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving data to labels.txt: {e}")
            print(f"Error saving data to labels.txt: {e}")
            # Do NOT close here for 'cancel' - let user see the error and decide

    def open_adder(self):
        dialog = AdderDialog(self)
        if dialog.exec():
            label, source = dialog.get_data()
            if label and source:
                row = self.table.rowCount()
                self.table.insertRow(row)
                self.table.setItem(row, 0, QTableWidgetItem(label))
                self.table.setItem(row, 1, QTableWidgetItem(source))
                self.data[label] = source # Update self.data when adding a new row
                self.table.resizeColumnsToContents()


    def get_subdirectory(self, folder_path):
        subdirectory_info = []
        if not os.path.isdir(folder_path):
            print(f"Error: '{folder_path}' is not a valid directory.")
            return subdirectory_info

        try:
            items_in_folder = os.listdir(folder_path)
            for item_name in items_in_folder:
                item_path = os.path.join(folder_path, item_name)
                if os.path.isdir(item_path):
                    subdirectory_info.append((item_path, item_name))
        except OSError as e:
            print(f"Error accessing folder '{folder_path}': {e}")
            return []

        return subdirectory_info


if __name__ == "__main__":
    app = QApplication([])
    window = FormAdder()
    window.show()
    app.exec()