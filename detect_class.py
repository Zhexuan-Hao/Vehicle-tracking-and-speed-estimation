from config_ui import *
import time
import json
import cv2
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from utils.general import non_max_suppression

def displayImg_out(mainwin, img):
        img = mainwin.padding(img)
        RGBImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        img_out = QImage(RGBImg, RGBImg.shape[1], RGBImg.shape[0], QImage.Format_RGBA8888)
        img_out = QPixmap(img_out)
        # img_out = img_out.scaledToWidth(self.mainwin.labelsize[1])
        img_out = mainwin.resizeImg(img_out)
        mainwin.label_out.setPixmap(img_out)
def displayImg_in(mainwin, img):
    img = mainwin.padding(img)
    RGBImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    img_out = QImage(RGBImg, RGBImg.shape[1], RGBImg.shape[0], QImage.Format_RGBA8888)
    img_out = QPixmap(img_out)
    # img_out = img_out.scaledToWidth(self.mainwin.labelsize[1])
    img_out = mainwin.resizeImg(img_out)
    mainwin.label_in.setPixmap(img_out)


def flow_detect(predict_class, url, save_vid, save_path):
    cap = cv2.VideoCapture(url)
    fps = int(cap.get(5))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if save_vid:
        fourcc = cv2.VideoWriter_fourcc(
            'm', 'p', '4', 'v')  # opencv3.0
        videoWriter = cv2.VideoWriter(
            save_path + '/output_video.mp4', fourcc, fps, (frame_width, frame_height))
    else:
        videoWriter = None

    while cap.isOpened():
        _, frame = cap.read()
        if frame is None: break

        # 帧转化为tensor
        img = img_transfer(frame)
        displayImg_in(predict_class.mainwin, frame)
        # 送入模型得到推理的boxes
        pred = p.model(img, augment=p.augment)[0]
        # nms处理，用于消除多余的框
        pred = non_max_suppression(pred, p.conf_thres, p.iou_thres, classes=p.classes, agnostic=p.agnostic_nms)
        boxes = get_boxes(pred, img, frame.shape, p.names)
        output_im = frame
        if len(boxes) > 0:
            # 从reid模型返回箱子们的id
            list_bboxs = track_update(boxes, frame, p.deepsort)
            # 把字刻在石头上
            output_im = draw_bboxes(frame, list_bboxs, predict_class,line_thickness=None) 
        if videoWriter is not None:
            videoWriter.write(output_im)
        displayImg_out(predict_class.mainwin, output_im)
        if predict_class.mainwin.stop == 1:
            break
    cap.release()
    if videoWriter is not None:
        videoWriter.release()

class PredictClass():
    def __init__(self, mainwin):
        self.predict_info_show = ''
        self.mainwin = mainwin

    def run(self, source):
        flow_detect(self, source, self.mainwin.save_img, self.mainwin.save_path)

    def displayInfo(self, info):
    # self.mainwin.predict_info_plainTextEdit.appendPlainText(info)
        self.predict_info_show = info

if __name__ == '__main__':
    flow_detect(0)
