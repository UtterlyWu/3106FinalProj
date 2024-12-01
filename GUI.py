import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget
from Compare import Comparer


class SimpleApp(QMainWindow):
    def __init__(self):
        self.model = Comparer()
        super().__init__()
        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        layout = QVBoxLayout()

        self.label1 = QLabel("Person 1 says...")
        layout.addWidget(self.label1)

        self.first_person = QLineEdit(self)
        layout.addWidget(self.first_person)

        self.label2 = QLabel("Person 2 says...")
        layout.addWidget(self.label2)

        self.second_person = QLineEdit(self)
        layout.addWidget(self.second_person)

        self.button = QPushButton("Do they agree?", self)
        self.button.clicked.connect(self.on_button_click)
        layout.addWidget(self.button)

        main_widget.setLayout(layout)

        self.result = QLabel("")
        layout.addWidget(self.result)

        self.setWindowTitle("Agreement Analyzer")
        self.setGeometry(100, 100, 400, 200)

    def on_button_click(self):
        pred = self.model.compare(self.first_person.text(), self.second_person.text())

        if(pred==0):
            self.result.setText("These people do not agree.")
            print("Disagree")
        elif(pred==2):
            self.result.setText("These people agree.")
            print("agree")
        #text = self.text_input.text()
        #self.label.setText(f"You entered: {text}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = SimpleApp()
    main_window.show()
    sys.exit(app.exec_())
