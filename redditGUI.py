import asyncio
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem
import subprocess
import os
import pandas as pd
from Compare import Comparer


class ResultWindow(QMainWindow):
    def __init__(self):
        self.comparer = Comparer()
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

        tuples = read_csv()
        index = 0
        for a,b in tuples:
            if type(a) is str and type(b) is str:
                self.table.setRowCount(index+1)
                pred = self.comparer.compare(a, b)
                if pred==0:
                    self.table.setItem(index,0,QTableWidgetItem("Not Agree"))
                else:
                    self.table.setItem(index,0,QTableWidgetItem("Agree"))
                self.table.setItem(index,1,QTableWidgetItem(a))
                self.table.setItem(index,2,QTableWidgetItem(b))
                index = index+1

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
        url = self.url.text() 
        print(f"URL: {url}")
        try:
            asyncio.run(run_scraper(url))
        
        except Exception as E:
            print("Invalid url")
            print(E)
        
        
        self.result_window = ResultWindow()
        self.result_window.show()

        # Example data to populate the table


async def run_scraper(url):
    js_script_path = "Reddit_Scraper/Scraper.js"
    output_csv = "output.csv"
    # try:
    # Run the Node.js script asynchronously and wait for it to complete
    process = await asyncio.create_subprocess_exec(
        "node", js_script_path, url,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()

    if process.returncode == 0:
        print(f"Scraper ran successfully for URL: {url}")
        print(stdout.decode())
        
        # Check if the CSV file exists
        if os.path.exists(output_csv):
            print(f"Output CSV located at: {os.path.abspath(output_csv)}")
        else:
            print("Error: Output CSV not found!")
    else:
        print(f"Scraper failed for URL: {url}")
        print(stderr.decode())
    # except Exception as e:
    #     print("An error occurred while running the scraper:")
    #     print(e)

def read_csv():
    # Read the CSV file
    df = pd.read_csv("output.csv")
    # Convert each row into a tuple
    tuples = [tuple(row) for row in df.values]
    return tuples


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = SimpleApp()
    main_window.show()
    sys.exit(app.exec_())
    # app = QApplication(sys.argv)
    # main_window = SimpleApp()
    # main_window.show()
    # sys.exit(app.exec_())

#https://www.reddit.com/r/unpopularopinion/comments/15ihq2x/burritos_suck_great_in_theory_terrible_in_reality/