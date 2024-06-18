import sys
from PyQt5.QtWidgets import QApplication, QMainWindow

import function  # 第一处

# （更改所有file_name 为之前生成的界面py的文件名，如name，不加.py）
if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = function.Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())