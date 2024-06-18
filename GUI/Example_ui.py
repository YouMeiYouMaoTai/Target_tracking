from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
# from Function import *

class Ui_MainWindow(QWidget):
    def __init__(self):
        QMainWindow.__init__(self)
        self.img_path = ""

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1300, 750)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(550, 260, 150, 30))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(550, 500, 150, 30))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.label1 = QtWidgets.QLabel(self.centralwidget)
        self.label1.setGeometry(QtCore.QRect(20, 170, 368, 448))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label1.setFont(font)
        self.label1.setObjectName("label1")
        self.label2 = QtWidgets.QLabel(self.centralwidget)
        self.label2.setGeometry(QtCore.QRect(900, 170, 368, 448))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label2.setFont(font)
        self.label2.setObjectName("label2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(500, 20, 250, 30))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(460, 60, 311, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setObjectName("label")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1300, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.pushButton.clicked.connect(lambda: self.openimage())
        self.pushButton_2.clicked.connect(lambda: self.recognition())
        self.pushButton_3.clicked.connect(lambda: self.recognition_rate())

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "HelloWorld!"))
        self.pushButton.setText(_translate("MainWindow", "选择图片"))
        self.pushButton_2.setText(_translate("MainWindow", "开始识别"))
        self.label1.setText(_translate("MainWindow", "1"))
        self.label2.setText(_translate("MainWindow", "2"))
        self.pushButton_3.setText(_translate("MainWindow", "统计识别率和检测率"))
        self.label.setText(_translate("MainWindow", "识别率"))

    # 打开图片文件
    def openimage(self):
        fname = QFileDialog.getOpenFileName(self, '选择图片', '', 'Image files(*.bmp)')
        self.img_path = fname[0]
        print("\n待识别图片路径：", self.img_path)
        img = QtGui.QPixmap(fname[0]).scaled(self.label1.width(), self.label1.height())
        self.label1.setPixmap(img)

    # 识别图片
    def recognition(self):
        result = result_img_path(self.img_path)
        print("识别结果图片路径：", result)
        if result:
            img = QtGui.QPixmap(result).scaled(self.label2.width(), self.label2.height())
            self.label2.setPixmap(QPixmap(img))

    # 统计识别正确率
    def recognition_rate(self):
        disp()
        result_recognition = success_rate()
        result_check = Check_rate()
        self.label.setText("识别正确率：" + str(result_recognition) + "%" + "\n"
                           + "检测成功率：" + str(result_check) + "%")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())