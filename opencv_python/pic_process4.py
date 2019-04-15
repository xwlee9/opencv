import cv2 as cv
import numpy as np
from matplotlib import  pyplot as plt


""" 画直线 矩形 圆 椭圆 多边形 放置文字

cv.line(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> None
    pt1: 线段的起点。
    pt2: 线段的终点。
    color: 线段的颜色，通过一个Scalar对象定义。
    thickness: 线条的宽度。
    lineType: 线段的类型。可以取值8，4，和CV_AA，分别代表8邻接连接线，4邻接连接线和反锯齿连接线。默认值为8邻接。为了获得更好地效果可以选用CV_AA(采用了高斯滤波)。
    shift: 坐标点小数点位数。


cv.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> None
p1: 左上角的点
p2: 右下角的点


画点可以是半径很小点圆
cv.circle(img, center, radius, color[, thickness[, lineType[, shift]]]) -> None
center: 圆心
radius: 半径


cv.ellipse(img, center, axes, angle, startAngle, endAngle, color[, thickness[, lineType[, shift]]]) -> None
cv.ellipse(img, box, color[, thickness[, lineType]]) -> None
center:椭圆圆心
axes:轴长（长轴长度，短轴长度）
angle:椭圆在逆时针方向上的旋转角度
startAngle,endAngle: 椭圆弧从长轴按顺时针方向测得的起始和结束


polylines(img, pts, isClosed, color[, thickness[, lineType[, shift]]]) -> None
pts 点的集合    pts = np.array（[[10,5]，[20,30]，[70,20]，[50,10]]，np.int32）
isClosed 是否闭合


putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]]) -> None
text:写入的文本数据 'opencv'
org:位置坐标    (100,100)
fontFace:字体类型 font
fontScale:字体缩放


"""

""" 一个plot demo
pyplot.hist(x, bins=None, range=None,normed=False,color=None):
    绘制直方图，x表示类似的一维数组（也可以是图像）bin(箱子)的个数,也就是总共有几条条状图 range设置显示的范围,范围之外的将被舍弃
    normed 归一化 color 颜色
    The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.
"""

def plot_demo(img):
    plt.hist(img.ravel(),256, density = True)  # ravel 将多位数组降至1维
    plt.show('img')


""" 直方图
cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate ]])
对于一个可迭代的（iterable）/可遍历的对象（如列表、字符串），enumerate将其组成一个索引序列，利用它可以同时获得索引和值
"""
def img_hist(img):
    color = ('b','g','r')
    for i,color in enumerate(color):
        hist = cv.calcHist([img],[i],None,[256],[0,256])
        plt.plot(hist, color = color)
        plt.xlim([0,256])
    plt.show()

"""直方图(分别显示)


"""
def ImgHist(img,type):
    color = (255,255,255)
    windowName = 'Gray'
    if type == 31:
        color = (255,0,0)
        windowName = 'B Hist'
    elif type == 32:
        color = (0,255,0)
        windowName = 'G Hist'
    elif type == 33:
        color = (0,0,255)
        windowName = 'R Hist'
    # 1 image 2 [0] 3 mask None 4 256 5 0-255
    hist = cv.calcHist([img],[0],None,[256],[0.0,256])
    minV,maxV,minL,maxL = cv.minMaxLoc(hist)  # 提取最大值用作归一化
    histImg = np.zeros([256,256,3],np.uint8)  # 用作画直方图
    for h in range(256):                       # 遍历1-255个颜色空间
        intenNormal = int(hist[h]*256/maxV)   # 归一化到0-255
        cv.line(histImg,(h,255),(h,255-intenNormal),color) # 归一化到255 从第255开始显示
    cv.imshow(windowName,histImg)
    return histImg

def img_hist1(img):
    channels = cv.split(img)
    for i in range(0,3):
        ImgHist(channels[i],31+i)

"""  直方图均衡化 灰度
cv.equalizeHist(src) 均衡化
局部增强
"""
def equalHist_demo(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    dst = cv.equalizeHist(gray)
    cv.imshow('dst',dst)

"""  直方图均衡化 彩色

hint: 将三通道的图片分成单通道图， 分别进行均衡
"""
def equalHist_color_demo(img):
    b, g, r = cv.split(img)
    bH = cv.equalizeHist(b)
    gH = cv.equalizeHist(g)
    rH = cv.equalizeHist(r)
    dst = cv.merge((bH,gH,rH))
    cv.imshow('dst',dst)


""" 两个图片相似度
  OpenCv提供了5种对比直方图的方式：CORREL（相关）、CHISQR（卡方）、INTERSECT（相交）、
  BHATTACHARYYA、EMD（最小工作距离），其中CHISQR速度最快，EMD速度最慢且有诸多限制，但是EMD的效果最好。
巴氏距离    HISTCMP_BHATTACHARYYA
相关性      HISTCMP_CORREL
卡方       HISTCMP_CHISQR
"""
def create_bgr_hist(img):
    h,w,c = img.shape
    bgrHist = np.zeros([16*16*16,1],np.float32)
    bsize = 256/16                       # bar size 16
    for row in range(h):
        for col in range(w):
            b = img[row,col,0]
            g = img[row,col,1]
            r = img[row,col,2]
#            if b == 255:
#                print(g,r)
            index = np.int(b/bsize)*16*16 + np.int(g/bsize)*16 + np.int(r/bsize)
            # 255/16 < 16 max        15*16*16 + 15*16 + 15 < 16*16*16
#            if index >16*16*16:
#                print (index)
            bgrHist[np.int(index),0] = bgrHist[np.int(index),0] + 1

    return bgrHist

def hist_compare(img1,img2):
    hist1 = create_bgr_hist(img1)
    hist2 = create_bgr_hist(img2)
    # compareHist 比较类型是float32
    match1 = cv.compareHist(hist1,hist2,cv.HISTCMP_BHATTACHARYYA) # 巴式距离
    match2 = cv.compareHist(hist1,hist2,cv.HISTCMP_CORREL) # 相关性
    match3 = cv.compareHist(hist1,hist2,cv.HISTCMP_CHISQR) # 卡方
    print('巴氏距离: %s,相关性: %s，卡方: %s'%(match1,match2,match3))



""" HSV直方图 """

def hist2d_demo(img):
    hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    hist = cv.calcHist([hsv],[0,1],None,[180,256],[0,180,0,256])
    cv.imshow('hist2d_demo',hist)
    plt.imshow(hist,interpolation='nearest')
    plt.title('2d histogram')
    plt.show()


""" 反向投影

反向投影是一种记录给定图像中的像素点如何适应直方图模型像素分布的方式，就是首先计算某一特征的直方图模型，
然后使用模型去寻找图像中存在的特征。反向投影在某一位置的值就是原图对应位置像素值在原图像中的总数目。
BackProjection中存储的数值代表了测试图像中该像素属于皮肤区域的概率。

灰度图像的像素值下图
0 1 2 3
4 5 6 7
8 9 10 11
8 9 14 15
对图像进行直方图统计（bin指定的区间为[0,3)，[4,7)，[8,11)，[12,16)）如下所示：
Histogram=4 4 6 2
也就是说在[0,3)这个区间的像素值有4个，其它含义相同
根据上述的直方图进行反向投影，得到反向投影图像像素值如下：
Back_Projection=

4 4 4 4
4 4 4 4
6 6 6 6
6 6 2 2

cv2.normalize(src[, dst[, alpha[, beta[, norm_type[, dtype[, mask]]]]]]) → dst
    src:输入数组。
    dst:与SRC大小相同的输出数组。
    α:范数值在范围归一化的情况下归一化到较低的范围边界。
    β:上限范围在范围归一化的情况下；它不用于范数归一化。
    norm_type:
    NORM_MINMAX:数组的数值被平移或缩放到一个指定的范围，线性归一化。
    NORM_INF: 归一化数组的（切比雪夫距离）L∞范数(绝对值的最大值)
    NORM_L1 :  归一化数组的（曼哈顿距离）L1-范数(绝对值的和)
    NORM_L2: 归一化数组的(欧几里德距离)L2-范数

cv.calcBackProject(images,channels,hist,ranges,scale,dst)
    channels :用于计算反向投影的通道列表
    hist : 输入的直方图，直方图的bin可以是密集(dense)或稀疏(sparse)
    ranges : 直方图中每个维度bin的取值范围
    scale = 1 : 可选输出反向投影的比例因子

"""
def backProjection_demo(img1,img2):
    sample = np.zeros([180,256,3],np.uint8)
    sample[0:180, 0:256, :] = img1[50:230, 100:356, :]
    target =  img2
    roi_hsv = cv.cvtColor(sample,cv.COLOR_BGR2HSV)  #将region of interest转化为HSV图像
    target_hsv = cv.cvtColor(target,cv.COLOR_BGR2HSV)   #将target转化为HSV图像

    cv.imshow('sample',sample)
    # cv.imshow("target",target)
    roiHist = cv.calcHist([roi_hsv],[0,1],None,[30,30],[0,180,0,256])
    cv.normalize(roiHist,roiHist,0,255,cv.NORM_MINMAX)
    dst = cv.calcBackProject([target_hsv],[0,1],roiHist,[0,180,0,255],1) #反向投影
    cv.imshow('dst',dst)

"""  模版的匹配

在整个图像区域发现与给定子图像匹配的区域，模板匹配的工作方式是在待检测图像上从左到右，从上到下计算模板图象与重
叠子图像的匹配度，匹配度越大，两者越相同

cv.matchTemplate(image, templ, method[, result]) -> result

cv2.TM_SQDIFF_NORMED，cv2.TM_SQDIFF这两种，其输出结果矩阵里最小值是匹配程度最好的，
        其余四种都是矩阵里最大值是匹配程度最好的。
6种method
CV_TM_SQDIFF 平方差匹配法：该方法采用平方差来进行匹配；最好的匹配值为0；匹配越差，匹配值越大。
CV_TM_CCORR 相关匹配法：该方法采用乘法操作；数值越大表明匹配程度越好。
CV_TM_CCOEFF 相关系数匹配法：1表示完美的匹配；-1表示最差的匹配。
CV_TM_SQDIFF_NORMED 归一化平方差匹配法
CV_TM_CCORR_NORMED 归一化相关匹配法
CV_TM_CCOEFF_NORMED 归一化相关系数匹配法




 """

def template_demo(img1,img2):
    tpl = np.zeros([180,256,3],np.uint8)
    tpl[0:180, 0:256, :] = img1[50:230, 100:356, :]
    target = img2
    methods = [cv.TM_SQDIFF_NORMED,cv.TM_CCORR_NORMED,cv.TM_CCOEFF_NORMED]
    th , tw = tpl.shape[:2]
    for md in methods:
        print(md)
        result = cv.matchTemplate(target,tpl,md)
        min_val,max_val,min_loc,max_loc = cv.minMaxLoc(result)  # 寻找到图上最大值 最小值到坐标
        if md == cv.TM_SQDIFF_NORMED:
            tl = min_loc
        else:
            tl = max_loc
        br = (tl[0]+tw,tl[1]+th)
        cv.rectangle(target,tl,br,(0,0,255),2)
        cv.imshow('match'+np.str(md),target)

""" 阈值 二值化
cv.threshold(src, thresh, maxval, type[, dst]) -> retval, dst
有两个返回值，第一个retVal（得到的阈值值），第二个就是阈值化后的图像
type:
THRESH_BINARY（黑白二值）  # 小于thresh 变成0 大于thresh ==》255
THRESH_BINARY_INV（黑白二值反转）
THRESH_TRUNC （得到的图像为多像素值）   # 小于thresh 变成thresh 滤波
THRESH_TOZERO
THRESH_TOZERO_INV

cv.THRESH_OTSU
Otsu’s Binarization是一种基于直方图的二值化方法，非常适合于图像灰度直方图具有双峰的情况
1. 计算图像直方图；
2. 设定一阈值，把直方图强度大于阈值的像素分成一组，把小于阈值的像素分成另外一组；
3. 分别计算两组内的偏移数，并把偏移数相加；
4. 把0~255依照顺序多为阈值，重复1-3的步骤，直到得到最小偏移数，其所对应的值即为结果阈值。
此时需要多传入一个参数cv2.THRESH_OTSU，并且把阈值thresh设为0，算法会找到最优阈值，并作为第一个返回值ret返回。

"""
def threshold_demo(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret , binary = cv.threshold(gray,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)
    # ret , binary = cv.threshold(gray,100,255,cv.THRESH_BINARY)    # 小于100 变成0 大于100 ==》255
    # ret , binary = cv.threshold(gray,100,255,cv.THRESH_TRUNC)    # 小于100 变成100 滤波
    plt.hist(gray.ravel(),256)
    plt.show('gray')

    print('threshold:',ret)
    cv.imshow('binary',binary)


"""  阈值为均值 二值化 """
def threhold_mean_demo(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    h,w = gray.shape[:2]
    m = np.reshape(gray,[1,w*h])
    mean = m.sum()/(w*h)
    print('mean : ',mean)
    ret,binary = cv.threshold(gray,mean,255,cv.THRESH_BINARY)
    cv.imshow('binary',binary)


""" 自适应阈值 二值化
cv.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst]) -> dst

第一个原始图像
第二个像素值上限
第三个自适应方法Adaptive Method:
    cv2.ADAPTIVE_THRESH_MEAN_C ：领域内均值
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C ：领域内像素点加权和，权重为一个高斯窗口
第四个值的赋值方法：只有cv2.THRESH_BINARY 和cv2.THRESH_BINARY_INV
第五个Block size:规定领域大小（一个正方形的领域）
第六个常数C，阈值等于均值或者加权值减去这个常数（为0相当于阈值 就是求得领域内均值或者加权值）
这种方法理论上得到的效果更好，相当于在动态自适应的调整属于自己像素点的阈值，而不是整幅图像都用一个阈值。

"""
def threhold_adp_demo(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # dst = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,25,10)
    dst = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,25,10)
    cv.imshow('binary',dst)


""" 大图片的二值化
hint: 进行分割处理
"""
def threhold_big_demo(img):
    print(img.shape)
    cw = 256
    ch = 256
    h,w = img.shape[:2]
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    for row in range(0,h,ch):
        for col in range(0,w,cw):
            roi = gray[row:row+ch,col:col+cw]

            # # 使用全局阈值 可以使用一个if判断 空白图像过滤
            # dev = np.std(roi)
            # if dev <=15:
            #     gray[row:row+ch,col:col+cw] = 255
            # else:
            #     ret,dst = cv.threshold(roi,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)
            #     gray[row:row+ch,col:col+cw] = dst

            dst = cv.adaptiveThreshold(roi,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,121,20)
            gray[row:row+ch,col:col+cw] = dst
    cv.imshow("threhold_big_demo",gray)
"""  图像金字塔   高斯金字塔
金字塔进行图像融合

需要对同一图像的不同分辨率的子图像进行处理。创建创建一组图像，这些图像是具有不同分辨率的原始图像。
我们把这组图像叫做图像金字塔（简单来说就是同一图像的不同分辨率的子图集合）。

高斯金字塔的顶部是通过将底部图像中的连续的行和列去除得到的。顶部图像中的每个像素值等于下一层图像中5个像素的高斯加权平均值。
操作一次一个 MxN 的图像就变成了一个 M/2xN/2 的图像。

pyrDown(src[, dst[, dstsize[, borderType]]]) -> dst
从一个高分辨率大尺寸的图像向上构建一个金子塔 （尺寸变小，分辨率降低）。

pyrUp(src[, dst[, dstsize[, borderType]]]) -> dst
从一个低分辨率小尺寸的图像向下构建一个金子塔（尺寸变大，但分辨率不会增加）。
"""
def pyramid_demo(img):
    level = 5
    temp = img.copy()
    pyramid_images = []
    for i in range(level):
        dst = cv.pyrDown(temp)
        pyramid_images.append(dst)
        cv.imshow("pyramid_images"+str(i),dst)
        temp = dst.copy()
    return pyramid_images


""" 拉普拉斯金字塔
拉普拉斯金字塔可以有高斯金字塔计算得来
Li = Gi − PyrUp (G(i+1))   # G(i+1) ===> m/2 * n/2   PyrUp (G(i+1)) ===> m * n
拉普拉金字塔的图像看起来就像边界图，其中很多像素都是0。他们经常 被用在图像压缩中。

注意img的尺寸 因为涉及 pyrDown 除法  再pyrUp回去的时候会和原尺寸不同
"""
def lapalcian_demo(img):
    pyramid_images = pyramid_demo(img)
    level = len(pyramid_images)   # 3
    for i in range(level-1,-1,-1):
# i=2 ===>pyrimg[2] 最小的一层 pyrUp===> size为pyrimg[2]的size ===> 上一层image(pyrimg[1])- 本层image(pyrimg[2])
# i=1 ===>pyrimg[1] pyrUp===> size为pyrimg[0]的size ===> 上一层image(pyrimg[0])- 本层image(pyrimg[1])
# 执行 else语句
# i=0 ===>pyrimg[0] pyrUp===> size为原图的size ===> 上一层image(原图)- 本层image(pyrimg[0])
        if (i-1)<0:
            expand = cv.pyrUp(pyramid_images[i],dstsize=img.shape[:2])
            lpls = cv.subtract(img,expand)
            cv.imshow("lapalian_demo"+str(i),lpls)
        else:
            expand = cv.pyrUp(pyramid_images[i], dstsize=pyramid_images[i-1].shape[:2])
            lpls = cv.subtract(pyramid_images[i-1],expand)
            cv.imshow("lapalian_demo"+str(i),lpls)








src = cv.imread('image0.jpg',1)
src1 = cv.imread('image_changed.jpg',1)
src2 = cv.imread('avatar.jpeg',1)
# cv.namedWindow("image",cv.WINDOW_NORMAL)

t1 = cv.getTickCount()

# plot_demo(src)
# img_hist(src)
# img_hist1(src)
# equalHist_demo(src)
# equalHist_color_demo(src)
# hist_compare(src1,src)
# hist2d_demo(src)
# backProjection_demo(src,src1)
# template_demo(src, src1)
# threshold_demo(src)
# threhold_mean_demo(src)
# threhold_adp_demo(src)
# threhold_big_demo(src)
# pyramid_demo(src)

# lapalcian_demo(src2)

t2 = cv.getTickCount()
time = (t2-t1)/cv.getTickFrequency()
print('time : %s s'%(time))

cv.waitKey(0)
cv.destroyAllWindows()
