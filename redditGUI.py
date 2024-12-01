import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem


class ResultWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Scraped Data Analysis")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        # Create a QTableWidget with 3 columns
        self.table = QTableWidget(self)
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Index", "Text A", "Text B"])

        # Adjust column sizes
        self.table.setColumnWidth(0, 100)  # Smaller first column
        self.table.setColumnWidth(1, 300)  # Larger second column
        self.table.setColumnWidth(2, 300)  # Larger third column

        layout.addWidget(self.table)

        # Set central widget with layout
        main_widget = QWidget()
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

    def populate_table(self, data):
        """
        Populate the table with data.
        :param data: List of tuples, where each tuple represents a row (index, text_a, text_b)
        """
        self.table.setRowCount(len(data))
        for row_index, (index, text_a, text_b) in enumerate(data):
            self.table.setItem(row_index, 0, QTableWidgetItem(str(index)))  # Index column
            self.table.setItem(row_index, 1, QTableWidgetItem(text_a))  # Text A column
            self.table.setItem(row_index, 2, QTableWidgetItem(text_b))  # Text B column


class SimpleApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        layout = QVBoxLayout()

        self.label1 = QLabel("Reddit Scraper Agreement Analyzer")
        layout.addWidget(self.label1)

        self.label2 = QLabel("Reddit URL:")
        layout.addWidget(self.label2)

        self.url = QLineEdit(self)
        layout.addWidget(self.url)

        self.button = QPushButton("Scrape and Analyze", self)
        self.button.clicked.connect(self.open_result_window)
        layout.addWidget(self.button)

        main_widget.setLayout(layout)

        self.setWindowTitle("Agreement Analyzer")
        self.setGeometry(100, 100, 400, 200)

    def open_result_window(self):
        # Create and show the result window
        self.result_window = ResultWindow()
        self.result_window.show()

        # Example data to populate the table
        example_data = [
            (1, "Person 1 said something...", "Person 2 replied with..."),
            (2, "Another comment from Person 1", "Another reply from Person 2"),
        ]
        self.result_window.populate_table(example_data)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = SimpleApp()
    main_window.show()
    sys.exit(app.exec_())
