from unittest import result
from UI.ui import Ui_MainWindow
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import cv2
# from utils.datasets import IMG_FORMATS, VID_FORMATS
import detect_class
import os
import numpy as np

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff',

               'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']


class PredictThread(QThread):
    # 自定义信号对象。参数str就代表这个信号可以传一个字符串
    trigger = pyqtSignal(int)

    def __init__(self, mainwin):
        # 初始化函数
        # super().__init__()
        super(PredictThread, self).__init__()
        self.mainwin = mainwin

    def run(self):
        #重写线程执行的run函数
        self.mainwin.model_class.run(self.mainwin.source)
        #触发自定义信号
        self.trigger.emit(1)

class PrintResultThread(QThread):
    """
    打印结果的线程
    """
    result_message_trigger = pyqtSignal(str)

    def __init__(self, mainwin):
        super(PrintResultThread, self).__init__()
        self.mainwin = mainwin

    def run(self):
        # while True:
        
        self.result_message_trigger.emit(self.mainwin.model_class.predict_info_show)


class MainWin(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("基于YOLOv5和DeepSort下的多视角视频的车速估计")

        self.PB_import.clicked.connect(self.importMedia)
        self.PB_predict.clicked.connect(self.run)
        # self.PB_predict.setEnabled(False)
        self.PB_stop.clicked.connect(self.stopPredict)
        self.PB_resize.clicked.connect(self.resize_label)
        self.stop = 0
        self.model_class = detect_class.PredictClass(self) 

        self.save_img = False
        self.save_path = ''
        self.CB_save_img.setChecked(False)  # 默认不保存
        self.PB_save_path.setEnabled(False)
        self.CB_save_img.stateChanged.connect(self.checkboxState)
        self.PB_save_path.clicked.connect(self.savepathSelect)
        
        self.classes = []

        self.source = None

        self.printResult = PrintResultThread(self)
        self.predictThread = PredictThread(self)
        # self.resultThread.trigger_result.connect(self.isdone_result)
        self.printResult.result_message_trigger.connect(self.result_info)
        self.predictThread.trigger.connect(self.isdone)

        self.timer_info = QTimer()
        self.timer_info.start(100)
        self.timer_info.timeout.connect(self.startprintresult)

    def checkboxState(self):
        if self.CB_save_img.isChecked():
            self.save_img  =True
            self.PB_save_path.setEnabled(True)
        else:
            self.save_img  =False
            self.PB_save_path.setEnabled(False)

    def savepathSelect(self):
        p = QFileDialog.getExistingDirectory(self, "选择路径", ".")
        if os.path.isdir(p):
            self.save_path = p
        else:
            self.predict_info_plainTextEdit.appendHtml('<font color=red>请选择文件夹...</font>')
            self.save_path = ''
    
    def checkboxState(self):
        if self.CB_save_img.isChecked():
            self.save_img  =True
            self.PB_save_path.setEnabled(True)
        else:
            self.save_img  =False
            self.PB_save_path.setEnabled(False)

    def startprintresult(self):
        self.printResult.start()


    def resizeImg(self, image):
        width = image.width()  ##获取图片宽度
        height = image.height() ##获取图片高度
        if width / self.labelsize[1] >= height / self.labelsize[0]: ##比较图片宽度与label宽度之比和图片高度与label高度之比
            ratio = width / self.labelsize[1]
        else:
            ratio = height / self.labelsize[0]
        new_width = width / ratio  ##定义新图片的宽和高
        new_height = height / ratio
        new_img = image.scaled(new_width, new_height)##调整图片尺寸
        return new_img

    def padding(self, image):
        width = image.shape[1]
        height = image.shape[0]
        target_ratio = self.labelsize[0]/self.labelsize[1] # h/w
        now_ratio = height/width
        if target_ratio>now_ratio:
            # padding h
            new_h = int(target_ratio*width)
            padding_image = np.ones([int((new_h-height)/2), width, 3], np.uint8)*255
            new_img = cv2.vconcat([padding_image, image, padding_image])
        else:
            # padding w
            new_w = int(height/target_ratio)
            padding_image = np.ones([height, int((new_w-width)/2), 3], np.uint8)*255
            new_img = cv2.hconcat([padding_image, image, padding_image])
        return new_img

    def resize_label(self):
        self.labelsize = [self.label_in.height(), self.label_in.width()]
        img_in = self.label_in.pixmap()
        img_out = self.label_out.pixmap()
        try:
            img_in = self.resizeImg(img_in)
        except:
            return
        else:
            self.label_in.setPixmap(img_in)

        try:
            img_out = self.resizeImg(img_out)
        except:
            return
        else:    
            self.label_out.setPixmap(img_out)


    def importMedia(self):
        self.labelsize = [self.label_in.height(), self.label_in.width()]
        if self.RB_camera.isChecked():
            self.source = 0
            # self.importImg('')
            # self.PB_import.setEnabled(False)
            self.predict_info_plainTextEdit.appendHtml('<font color=green>点击预测后将自动打开摄像头...</font>')
            # self.importVideo(self.source)
            # print(source)
        elif self.RB_img.isChecked():
            fname, _ = QFileDialog.getOpenFileName(self, "打开文件", ".")
            # print(fname)
            if fname.split('.')[-1].lower() in (VID_FORMATS):
                self.importImg(fname)
                self.source = fname          
            else:
                self.predict_info_plainTextEdit.appendHtml('<font color=red>不支持该类型文件...</font>')
        else:
            self.predict_info_plainTextEdit.appendHtml('<font color=red>请选择检测源类型...</font>')
    
    def importImg(self, file_name):
        if file_name.split('.')[-1].lower() in VID_FORMATS:
            cap = cv2.VideoCapture(file_name)
            if cap.isOpened():
                # self.video = True
                ret, img_in = cap.read()
                if ret:
                    img_in = self.padding(img_in)
                    img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGBA)
                    # padding
                    img_in = QImage(img_in, img_in.shape[1], img_in.shape[0], QImage.Format_RGBA8888)
                    img_in = QPixmap(img_in)
            cap.release()
        elif file_name.split('.')[-1].lower() in IMG_FORMATS:
            # self.video = False
            img_in = cv2.imread(file_name)
            img_in = self.padding(img_in)
            img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGBA)
            img_in = QImage(img_in, img_in.shape[1], img_in.shape[0], QImage.Format_RGBA8888)
            img_in = QPixmap(img_in)
        if img_in.isNull():
            # self.predict_info_plainTextEdit.appendHtml('<font color=red>打开失败...</font>')
            print('打开失败...')
            return
        # img_in = img_in.scaledToWidth(self.labelsize[1])
        img_in = self.resizeImg(img_in)
        self.label_in.setPixmap(img_in)

    def stopPredict(self):
        self.stop = 1

    def run(self):
        if self.source == None:
            # self.predict_info_plainTextEdit.appendHtml('<font color=red>请选择检测源...</font>')
            print('请选择检测源...')
            return
        else:
            self.predictThread.start()
            # self.canrun = 0
            self.PB_predict.setEnabled(False)
            # self.action_loadmodel.setEnabled(False)
            self.PB_stop.setEnabled(True)
            # self.model_class.run()

    def result_info(self,info_predict):
        if info_predict != '':
            self.predict_info_plainTextEdit.appendPlainText(info_predict)
            self.model_class.predict_info_show = ''

    def isdone(self, done):
        if done == 1:
            self.canrun = 1
            self.PB_predict.setEnabled(True)
            self.action_loadmodel.setEnabled(True)
            # self.PB_import.setEnabled(True)
            self.PB_stop.setEnabled(False)
            self.stop = 0
            self.predictThread.quit()





if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = MainWin()
    main.show()
    sys.exit(app.exec_())
