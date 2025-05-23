# PIXSTREAM

使用像素图传输文件，在无法从远程连接或虚拟机直接复制文件的情况下，可以通过将数据转为字节流，在将字节流转为RGB图像，再通过屏幕截图从RGB图像解码数据，从而传输文件

适用条件：

- 较高质量显示远程画面
- 粘贴板能单向同步到远程

实现原理：

- 发送端将文件gzip压缩
- 将压缩后数据切分为多个chunk
- 将每个chunk的字节数组转为图像，一个RGB像素对于3字节数据
- 接收端通过截图，读取图像像素值重新转为字节数组
- 接收端通过粘贴板告知发送端是否接受成功以便重发

为什么不用二维码：二维码由于需要定位矫正解码等复杂操作通常只能达到2-3kb/s传输速率，PIXSTREAM可以实现600KB/s

## Requirements

```
pip install numpy opencv-python
pip install mss screeninfo
pip install pyperclip tqdm
pip install pywin32
```

## run

虚拟机端：

`python sender.py file_path`

客户端：

`python receiver.py`

## 遇到的问题

1. 由于远程连接画面传输存在图像压缩或者可能存在噪声等原因，导致预期显示像素值和直接截图获取像素值存在上下4左右偏差

   解决方法：通过md5校验数据，确保数据准确性，同时使用2倍bytes冗余表达，具体做法：将1byte=8bit分为左右4bit，4bit表示范围为0-15，通过8bit存储4bit信息实现一个范围表示同一个数：0映射到0-15, 1映射到16-32等

2. 由于存在水印等部分遮挡，导致某些位置图像无法正确解析

   解决方法：通过接受失败数据重发，以及打乱数据顺序避免同一个数据总是被固定水印或者随机水印命中

## 可能存在的问题

1. 远程连接画面或虚拟机画面和截图画面存在偏移

   解决方法：通过在开始传输前通过特殊图案进行校准（截图，逐像素比对，类似模板匹配），计算画面偏移（TODO）



该项目下的所有资源仅供学习和研究使用。其旨在为学术和研究人员提供参考和资料，任何其他目的均不适用
