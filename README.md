# 一个基于DEEPSORT和YOLO5的无人机监控摄像头车辆监测系统

## 工程结构如下：

/data：工程所用到的pretrained模型。YOLO5m是YOLO5系列在精度和速度方面的平衡，但也可以换成别的
/deep_sort：DeepSort ReID模型代码库，来自官方实现
/model：YOLO模型相关库，来自官方实现
/plate_identity:车牌识别相关库，暂不启用
/utils：工程相关工具
111.mp4：显然这是个用来跑demo的视频
config.py：包括所有核心代码
main_video.py：项目的启动入口-从本地视频推理
main.py：项目的启动入口-从在线摄像头推理

## 环境配置
Python版本3.7, CUDA 11。
此项目所需的包都在requirements.txt中，直接pip install -r requirements.txt即可。

最小应具有4GB图形显存的显卡。

## 运行
从摄像头推理：
python main.py
从本地视频推理：
python main_video.py

其他注解于代码中。

## 理解代码
看懂config.py里的draw_boxes方法。先从draw_boxes开始看，在过程中就能搞清楚别的函数作用。

测速算法要义：
首先定义屏幕上1cm等于现实中1m，这里是第一个可以调整的参数，在不同环境下适应时需要调整这个参数。
其次定义屏幕上1cm等于40个像素，这个参数应根据不同摄像头具体的像素数调整。
进一步的，我们计算车辆移动的路程（注意不是位移）。
假设目标一直不丢，摄像头运行在每秒30帧速度，这里也应该根据不同的摄像头/数据流调整，若有需要。
通过计算每个车的box数量，结合上步推算出车辆运行了多少秒。

下面计算真实世界的车辆速度，默认定义锚点为72km/h。转换为米/秒。即可计算出物体在真实世界中的速度。
这一部分请结合算法实现阅读，需要注意的是，因为视察效应，测速区间应为摄像头靠中的一块区域。应限制定义域为此范围以获取准确的测速结果。
在模拟中，此算法在参数调节得当的情况下对速度测试较为精准。

## 关于车牌识别
代码在plate_identity里，接口是传入一个照片（不需要裁剪ROI），返回车牌内容。因为没有合适的数据所以不启动。未来启用时要把接口实例化掉。
# Vehicle-tracking-and-speed-estimation
