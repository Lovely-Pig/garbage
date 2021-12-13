# garbage

## 概述

垃圾分类的目标检测

## 运行方法

本项目比较耗GPU，所以租用了[openbayes](https://openbayes.com/)的GPU资源进行训练

1. 安装相关依赖

   ```bash
   $ cd garbage/
   $ pip install -r requirements.txt
   ```

2. 下载预训练模型`yolov3.weights`

   ```bash
   $ sh weights.sh
   ```

3. 使用自己的数据集训练模型

   ```bash
   $ python train.py
   ```

4. 执行检测任务

   ```bash
   # 对文件夹下的图片进行检测
   $ python detect.py

   # 摄像头实时检测
   $ python cam_detect.py
   ```

## 参考资料

- [PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)
- [yolo](https://pjreddie.com/darknet/yolo/)
- [yolov3-paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
- [openbayes](https://openbayes.com/)
