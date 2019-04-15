import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

"""分水岭算法
任何一副灰度图像都可以被看成拓扑平面，灰度值高的区域可以被看成是山峰，灰度值低的区域可以被看成是山谷。
我们向每一个山谷中灌不同颜色的水。随着水的位的升高，不同山谷的水就会相遇汇合，为了防止不同山谷的水汇合，
我们需要在水汇合的地方构建起堤坝。不停的灌水，不停的构建堤坝直到所有的山峰都被水淹没。我们构建好的堤坝就是对图像的分割。
水位不断上升，汇水盆地之间及其与背景之间的坝也越来越长。构筑水坝的目的是阻止盆地之间及其与背景之间的水汇聚。
该过程一直持续，直到到达水的最高水位。最终水坝就是我们希望的分割结果。这样，在两个区域之间就给出了连续的边界。

距离变换的基本含义是计算一个图像中非零像素点到最近的零像素点的距离，也就是到零像素点的最短距离。
1个最常见的距离变换算法就是通过连续的腐蚀操作来实现，腐蚀操作的停止条件是所有前景像素都被完全腐蚀。
这样根据腐蚀的先后顺序，我们就得到各个前景像素点到前景骨架像素点的距离。根据各个像素点的距离值，设置为不同的灰度值。完成了二值图像的距离变换

cv.distanceTransform(src, distanceType, maskSize)
第二个参数 0,1,2 分别表示
    CV_DIST_L1 : |x1 - x2| + |y1 - y2|
    CV_DIST_L2 : 欧式距离
    CV_DIST_C : MAX(|x1 - x2|,|y1 - y2|)
maskSize: 距离变换的大小

cv.connectedComponents(image, labels, connectivity, ltype)

实现流程:
1.使用 Otsu's 二值化
2.使用形态学中的开运算去除图像中的所有的白噪声
3.进行膨胀操作找到肯定不是硬币的区域
4.距离变换再加上合适的阈值，找到肯定是前景的区域(只需要对前景进行分割，而不需要将紧挨在一起的对象分开可以只用腐蚀)
5.两者相减为边界区域 即不确定区域
6.创建标签（一个与原图像大小相同，数据类型为in32的数组）并标记其中的区域。用connectedComponents()
    对已经确定分类的区域(无论是前景还是背景)使用不同的正整数标记，对不确定的区域使用0标记。
7.分水岭算法
"""

def water_shed_demo(img):
    # 二值化处理
    print(img.shape)
    blurred = cv.pyrMeanShiftFiltering(img, 10, 100)
    gray = cv.cvtColor(blurred,cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    cv.imshow('binary',binary)

    # morphology operation  noise removal
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    mb = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    cv.imshow('open',mb)

    # sure background area 利用膨胀操作 我们可以确定那些地方肯定是背景
    sure_bg = cv.dilate(mb, kernel, iterations = 3)
    cv.imshow('mor_opt', sure_bg)

    #distance transform
    dist = cv.distanceTransform(mb, cv.DIST_L2, 3) 
    dist_output = cv.normalize(dist, 0, 1.0, cv.NORM_MINMAX)  #归一化利于显示
    cv.imshow('distance_t', dist_output*100)

    # 前景
    ret, surface = cv.threshold(dist, dist.max()*0.6, 255, cv.THRESH_BINARY)
    cv.imshow('surface', surface)

    # Finding unknown region 合并标记图像
    surface_fg = np.uint8(surface)
    unknow = cv.subtract(sure_bg, surface_fg)

    # Marker labelling
    ret, markers = cv.connectedComponents(surface_fg)
    print(ret)


    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknow == 255] = 0
    # 分水岭算法 标签图像将会被修改，边界区域的标记将变为 -1.
    markers = cv.watershed(img, markers)

    img[markers == -1] = [0, 0, 255]
    cv.imshow('result', img)

""" 对象检测 Haar 分类器
以 Haar 特征分类器为基础的对象检测技术是一种非常有效的对象检测技术。它是基于机器学习的，
通过使用大量的正负样本图像训练得到一个 cascade_function，最后再用它来做对象检测。

算法需要大量的正样本图像(面部图像)和负样本图像(不含面部的图像)来训练分类器从其中提取特征。Haar特征会被使用。
    它们就像我们的卷积核。每一个特征是一个值，这个值等于黑色矩形中的像素值之后减去白色矩形中的像素值之和。
将每一个特征应用于所有的训练图像。对于每一个特征，找到它能够区分出正样本和负样本的最佳阈值。选取错误率最低的特征。
最终的分类器是弱分类器的加权和。成为弱分类器因为只是用这些分类器不足以对图像进行分类，与其他的分类器联合起来就是一个很强的分类器了。

级联分类器将这些特征分成不同组。在不同的分类阶段逐个使用。(通常前面很少的几个阶段使用较少的特征检测)。
如果一个窗口第一阶段的检测过不了就可以直接放弃后面的测试了，如果它通过了就进入第二阶段的检测。
如果一个窗口经过了所有的测试，那么这个窗口就被认为是面部区域。
"""
def face_detection_demo(img):
    # OpenCV 已经包含了很多已经训练 好的分类器，其中包括：面部，眼睛，微笑等。
    # 加载需要的 XML 分类器。
    face_cascade = cv.CascadeClassifier('/anaconda3/envs/tensorflow37/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml') 
    eye_cascade = cv.CascadeClassifier('/anaconda3/envs/tensorflow37/share/OpenCV/haarcascades/haarcascade_eye.xml')

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    # 返回面部所在的矩形区域 Rect（x,y,w,h）。 n个
    for (x,y,w,h) in faces:
        img = cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) 
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = img[y:y+h, x:x+w] 
        eyes = eye_cascade.detectMultiScale(roi_gray) 
        for (ex,ey,ew,eh) in eyes:
            cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv.imshow('img',img)

def video_demo():
    caputer = cv.VideoCapture(0)   
    while(True):
        ret, frame = caputer.read()
        frame = cv.flip(frame, 1)  # 图像的水平镜像反转
        face_detection_demo(frame)
        # cv.imshow('video', frame)
        c = cv.waitKey(25)   
        if c == 27:
            break


""" 图像修补 

使用坏点周围的像素取代坏点，这样看起来和周围像素就比较像
inpaint(src, inpaintMask, inpaintRadius, flags, dst)
flags:
cv.INPAINT_TELEA 基于快速行进算法的。算法从这个区域的边界开始向区域内部慢慢前进，首先填充区域边界像素。
    选取待修补像素周围的一个小的邻域，使用这个邻域内的归一化加权和更新待修复的像素值。权重的选择是非常重要的。
    对于靠近带修复点的像素点，靠近正常边界像素点和在轮廓上的像素点给予更高的权重。当一个像素被修复之后，使用快速行进算法（FMM）移动到下一个最近的像素。

cv.INPAINT_NS 基于流体动力学并使用偏微分方程。首先沿正常区域的边界向退化区域的前进(边界连续，所以退化区域非边界与正常区域的边界应该也是连续的)。
    通过匹配待修复区域中的梯度向量来延伸等光强线(isophotes，由灰度值相等的点练成的线)。通过填充颜色来使这个区域内的灰度值变化最小。
"""
def paint_demo(img):
    # 破坏图片
    for i in range(200,300):
        img[i,200] = (255,255,255)
        img[i,200+1] = (255,255,255)
        img[i,200-1] = (255,255,255)
    for i in range(150,250):
        img[250,i] = (255,255,255)
        img[250+1,i] = (255,255,255)
        img[250-1,i] = (255,255,255)
    cv.imshow('damaged',img)

    # 得到图片信息
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    paint = np.zeros((height,width,1),np.uint8)  # 灰度图像 

    for i in range(200,300):
        paint[i,200] = 255
        paint[i,200+1] = 255
        paint[i,200-1] = 255
    for i in range(150,250):
        paint[250,i] = 255
        paint[250+1,i] = 255
        paint[250-1,i] = 255
    cv.imshow('paint',paint)

    imgDst = cv.inpaint(img,paint,3,cv.INPAINT_TELEA)
    cv.imshow('image',imgDst)
"""视频===>图片
"""
def video2picture_demo():
    caputer = cv.VideoCapture("1.mp4")
    isOpened = caputer.isOpened # 判断是否打开
    print(isOpened)
    fps = caputer.get(cv.CAP_PROP_FPS)
    w = int(caputer.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(caputer.get(cv.CAP_PROP_FRAME_HEIGHT))
    print (fps, w, h)
    i = 0
    while(isOpened):
        if i == 10:
            break
        else:
            i = i + 1
        ret, frame = caputer.read()
        fileName = 'video2pic'+str(i)+'.jpg'
        print (fileName)
        if ret == True: 
            cv.imwrite(fileName,frame,[cv.IMWRITE_JPEG_QUALITY,100])
    print ('end!!!')


"""视频写入
cv.VideoWriter([filename, fourcc, fps, frameSize[, isColor]]) -> <VideoWriter object>
写入对象 1 file name 2 编码器 3 帧率 4 size(size为宽*高 也就是列*行)
"""
def picture2video_demo():
    img = cv.imread('video2pic1.jpg',1)
    imgInfo = img.shape
    size = (imgInfo[1],imgInfo[0])   # (1280, 720)
    print(size) 
    
    videoWrite = cv.VideoWriter('2.mp4',-1,5,size)
    for i in range(1,11):
        fileName = 'video2pic'+str(i)+'.jpg'
        img = cv.imread(fileName)
        videoWrite.write(img)# 写入方法 1 jpg data
print('end!')

src = cv.imread('water_coins.jpg',1)
src1 = cv.imread('image0.jpg',1)
cv.namedWindow('img',cv.WINDOW_AUTOSIZE)
t1 = cv.getTickCount()

# water_shed_demo(src)
# face_detection_demo(src)
# video_demo()
# paint_demo(src1)
# video2picture_demo()

# picture2video_demo()

t2 = cv.getTickCount()
time = (t2-t1)/cv.getTickFrequency()
print('time : %s s'%(time))
cv.waitKey(0)
cv.destroyAllWindows()
