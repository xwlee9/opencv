import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt



""" 图像梯度 Sobel算子
梯度简单来说就是求导。OpenCV 提供了三种不同的梯度滤波器，或者说高通滤波器：Sobel，Scharr和Laplacian。
Sobel算子是高斯平滑与微分操作的结合体，抗噪声能力很好。可以设定求导的方向xorder或yorder。还可以设定使卷积核的大小ksize
如果 ksize=-1，会使用3x3的 Scharr 滤波器，它的的效果要 比3x3的 Sobel 滤波器好

Sobel，Scharr 其实就是求一阶或二阶导数。Scharr是对Sobel（使用小的卷积核求解求解梯度角度时）的优化。

cv.Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]]) -> dst
cv.Scharr(src, ddepth, dx, dy[, dst[, scale[, delta[, borderType]]]]) -> dst

src:是需要处理的图像；
ddepth:是图像的深度，-1表示采用的是与原图像相同的深度。目标图像的深度必须大于等于原图像的深度 设定为CV_16S避免外溢
dx,dy: 表示的是求导的阶数，一般为0、1、2。这里1表示对X或Y求偏导（差分），0表示不对X或Y求导（差分）。
其后是可选的参数：
dst不用解释了；
ksize是Sobel算子的大小，必须为1、3、5、7。
scale是缩放导数的比例常数，默认情况下没有伸缩系数；
delta是一个可选的增量，将会加到最终的dst中，同样，默认情况下没有额外的值加到dst中；
borderType是判断图像边界的模式。这个参数默认值为cv2.BORDER_DEFAULT。

Sobel函数求完导数后会有负值，还有会大于255的值。而原图像是uint8，即8位无符号数，所以Sobel建立的图像位数不够，会有截断。
因此要使用16位有符号的数据类型，即cv2.CV_16S。 也可以是cv2.CV_32F或者cv2.CV_64F表示32/64位浮点数即64float。

在经过处理后，用convertScaleAbs()函数将其转回原来的uint8形式。
cv.convertScaleAbs(src[, dst[, alpha[, beta]]]) -> dst
alpha是伸缩系数，beta是加到结果上的一个值。

最后要用cv.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]]) -> dst函数将其组合起来。
"""
def sober_demo(img):
    grad_x = cv.Sobel(img,cv.CV_32F,1,0)
    grad_y = cv.Sobel(img,cv.CV_32F,0,1)
    # grad_x = cv.Scharr(img, cv.CV_32F, 1, 0)
    # grad_y = cv.Scharr(img, cv.CV_32F, 0, 1)
    gradx = cv.convertScaleAbs(grad_x)
    grady = cv.convertScaleAbs(grad_y)
    gradxy = cv.addWeighted(gradx,0.5,grady,0.5,0)
    cv.imshow('dst', gradxy)



""" 图像梯度 Laplacian算子
Laplacian算子 是求二阶导数 图像中的边缘区域，像素值会发生“跳跃”，对这些像素求导，在其一阶导数在边缘位置为极值，
就是Sobel算子使用的原理——极值处就是边缘。如果对像素值求二阶导数，会发现边缘处的导数值为0
opencv在计算拉普拉斯算子时直接调用Sobel算子。
∆src = src(对x二阶偏导) + src(对y二阶偏导)

cv.Laplacian(src, ddepth[, dst[, ksize[, scale[, delta[, borderType]]]]]) -> dst
cv.filter2D(src, ddepth, kernel[, dst[, anchor[, delta[, borderType]]]]) -> dst
"""
def laplacian_demo(img):
    # dst = cv.Laplacian(img, cv.CV_32F)
    kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    kernel1 = np.array([[1,1,1],[1,-8,1],[1,1,1]])
    dst = cv.filter2D(img,cv.CV_32F,kernel=kernel)
    dst1 = cv.filter2D(img,cv.CV_32F,kernel=kernel1)
    lpls = cv.convertScaleAbs(dst)
    lpls1 = cv.convertScaleAbs(dst1)
    cv.imshow('lpls',lpls)
    cv.imshow('lpls1',lpls1)

"""边缘检测 canny
对平滑后的图像使用Sobel算子计算水平方向和竖直方向的一阶导数（图像梯度）Gx和Gy。梯度的方向一般总是与边界垂直。
要确定那些边界才是真正的边界。设置两个阈值： minVal 和 maxVal。
当图像的灰度梯度高于maxVal时被认为是真的边界，那些低于minVal的边界会被抛弃。
如果介于两者之间的话，就要看这个点是否与某个被确定为真正的边界点相连，如果是就认为它也是边界点，如果不是就抛弃。

cv.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]]) -> edges
image：输入图像
threshold1, threshold2是 minVal 和 maxVal。
apertureSize：计算图像梯度的 Sobel 卷积核的大小，默认值为 3。
L2gradient它可以用来设定求梯度大小的方程，默认False。True为Edge_Gradient(G) =sqr(Gx^2 + Gy^2) False为|Gx^2| + |Gy^2|
"""

def canny_demo(img):
    blurred = cv.GaussianBlur(img, (3,3), 1,1) # 噪声敏感 高斯去噪声
    gray = cv.cvtColor(blurred,cv.COLOR_BGR2GRAY)

    xgrad = cv.Sobel(gray,cv.CV_16SC1,1,0)
    ygrad = cv.Sobel(gray,cv.CV_16SC1,0,1)

    edge_output = cv.Canny(gray,50,150)
    edge_output1 = cv.Canny(xgrad,ygrad,50,150)

    cv.imshow('edge_output',edge_output)
    # cv.imshow("edge_output1",edge_output1) # 也可以传入sober算子

    dst = cv.bitwise_and(img,img,mask=edge_output)  # 掩模运算 mask只有边缘信息 其余为0 ===> 抠出彩色图片的边缘
    cv.imshow('color_edge',dst)


"""霍夫变换
霍夫变换在检测各种形状的的技术中非常流行，如果要检测的形状可以用数学表达式写出，就可以是使用霍夫变换检测它。
即使要检测的形状存在一点破坏或者扭曲也可以使用。

一条直线可以用数学表达式 y = mx + c 或者 ρ = x cos θ + y sin θ 表示。
ρ 是从原点到直线的垂直距离，θ 是直线的垂线与横轴顺时针方向的夹角

每一条直线都可以用(ρ, θ)表示。所以首先创建一个 2D 数组（累加器），初始化累加器，所有的值都为0。
行表示 ρ，列表示θ。这个数组的大小决定了最后结果的准确性。如果希望角度精确到1度，你就需要180列。
对于ρ，最大值为图片对角线的距离。有一个大小为 100x100 的直线位于图像的中央。取直线上的第一个点，知道此处的（x，y）值。
把x和y带入上边的方程组，然后遍历θ的取值：0，1，2，3，. . .，180。分别求出与其对应的ρ的值，这样就得到一系列（ρ, θ）的数值对。
如果这个数值对在累加器中也存在相应的位置，就在这个位置上加1。所以现在累加器中的（50，90）=1。
（一个点可能存在与多条直线中，所以对于直线上的每一个点可能是累加器中的多个值同时加1。
继续取第二个点，重复过程，更新累加器的值。

cv.HoughLines(image, rho, theta, threshold[, lines[, srn[, stn]]]) -> lines
image:是一个二值化图像，所以在进行霍夫变换之前要首先进行二值化，或者进行 Canny 边缘检测。
rho, theta:分别代表 ρ 和 θ 的精确度。
threshold:阈值，只有累加其中的值高于阈值时才被认为是一条直线，也可以把它看成能检测到的直线的最短长度(以像素点为单位)。
返回值就是（ρ, θ）
"""
def hough_detection_demo(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray,50,150,apertureSize = 3)

    lines = cv.HoughLines (edges,1,np.pi/180,150)
    print(lines.shape)   #(n,1,2)

    # for a,b in lines[0]:
    #     print(a,b)

    for line in lines:
        print (line)    # 打印n个 （ρ, θ）
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000*(-b)) # x1左下方向  ρ*cosθ - N*sinθ
        y1 = int(y0 + 1000*(a))  # y1左下方向  ρ*sinθ + N*sinθ
        x2 = int(x0 - 1000*(-b)) # x1右上方向  ρ*cosθ + N*sinθ
        y2 = int(y0 - 1000*(a))  # y1右上方向  ρ*sinθ - N*sinθ

        cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)


    cv.imshow('hough_line', img)

"""霍夫变换
Probabilistic_Hough_Transform 是对霍夫变换的一种优化。它不会对每一个点都进行计算，
而是从一幅图像中随机选取一个点集进行计算，对于直线检测来说这已经足够了,因为总点数少了，所以我们要降低阈值

HoughLinesP(image, rho, theta, threshold[, lines[, minLineLength[, maxLineGap]]]) -> lines
minLineLength - 线的最短长度。比这个短的线都会被忽略。
MaxLineGap - 两条线段之间的最大间隔，如果小于此值，这两条直线 就被看成是一条直线。
函数的返回值就是直线的起点和终点。
"""

def hough_detection_possible_demo(img): # 可以是小的线段
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray,50,150,apertureSize = 3)
    lines = cv.HoughLinesP (edges,1,np.pi/180,80,minLineLength=50,maxLineGap=10)
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    cv.imshow('hough_detection_possible_demo',img)


"""霍夫圆的检测
一般线进行去噪声 对噪声敏感
圆形的数学表达式为 (x − xcenter )^2 + (y − ycenter )^2 = r^2,其中（xcenter,ycenter）为圆心的坐标，r为圆的直径。
==>1个圆环需要3个参数来确定。即圆环霍夫变换的累加器必须是3维的，这样效率会很低。所以OpenCV用霍夫梯度法，使用边界的梯度信息。

cv.HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles

image:输入矩阵
method:cv2.HOUGH_GRADIENT 也就是霍夫圆检测，梯度法
dp:计数器的分辨率图像像素分辨率与参数空间分辨率的比值
    图像分辨率与累加器分辨率的比值，把参数空间认为是一个累加器，里面存储是经过像素点的数量
    dp=1，则参数空间与图像像素空间（分辨率）一样大，dp=2，参数空间的分辨率只有像素空间的一半大
minDist 圆心之间最小距离，如果距离太小，会产生很多相交的圆，如果距离太大，则会漏掉正确的圆
param1 canny检测的双阈值中的高阈值，低阈值是它的一半
param2 最小投票数（基于圆心的投票数）
minRadius 需要检测院的最小半径
maxRadius 需要检测院的最大半径
返回N个圆的信息储存在1xN×3的ndarray。1*N个[rx,ry,R]
"""
def hough_detection_circles_demo(img):
    dst = cv.pyrMeanShiftFiltering(img,10,100) # 对噪声敏感
    cimage = cv.cvtColor(dst,cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(cimage,cv.HOUGH_GRADIENT,1,20,param1 = 50,param2= 30,minRadius =0,maxRadius =0)
    circles = np.uint16(np.around(circles))
    print(circles.shape)
    for i in circles[0,:]:
        # draw the outer circle
        cv.circle(img,(i[0],i[1]),i[2],(0,0,255),2)
        # draw the center of the circle
        cv.circle(img,(i[0],i[1]),2,(255,0,0),2)
    cv.imshow('circles',img)
"""轮廓检测
轮廓可以简单认为成将连续的点（连着边界）连在一起的曲线，具有相同的颜色或者灰度。轮廓在形状分析和物体的检测和识别中很有用。
通常要使用二值化图像。在寻找轮廓之前，要进行阈值化处理或者 Canny 边界检测。查找轮廓函数会改变原始图像。
cv.findContours(image, mode, method[, contours[, hierarchy[, offset]]]) -> contours, hierarchy
image:寻找轮廓的图像
mode:表示轮廓的检索模式，有四种:
    cv2.RETR_EXTERNAL表示只检测外轮廓
    cv2.RETR_LIST检测的轮廓不建立等级关系
    cv2.RETR_CCOMP建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。
                    如果内孔内还有一个连通物体，这个物体的边界也在顶层。
    cv2.RETR_TREE建立一个等级树结构的轮廓。
method:为轮廓的近似办法
    cv2.CHAIN_APPROX_NONE 存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max(abs(x1-x2),abs(y2-y1))==1
    cv2.CHAIN_APPROX_SIMPLE 压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
    cv2.CHAIN_APPROX_TC89_L1    使用teh-Chinl chain 近似算法
    cv2.CHAIN_APPROX_TC89_KCOS  使用teh-Chinl chain 近似算法
返回值有三个，第一个是图像，第二个是轮廓，第三个是（轮廓的）层析结构。轮廓（第二个返回值）是一个 Python 列表，
    其中存储这图像中的所有轮廓。每一个轮廓都是一个 Numpy 数组，包含对象边界点（x，y）的坐标。



cv.drawContours(image, contours, contourIdx, color[, thickness[, lineType[, hierarchy[, maxLevel[, offset]]]]]) -> None
image:是输入图像
contours:是轮廓,一个Python列表。
contourIdx轮廓的索引(在绘制独立轮廓是很有用，当设置为-1时绘制所有轮廓)
"""

def edge_detection(img):
    blurred = cv.GaussianBlur(img,(3,3),0)
    gray = cv.cvtColor(blurred,cv.COLOR_BGR2GRAY)

    edge_out = cv.Canny(gray, 50, 100)
    return edge_out

def contours_demo(img):
    binary = edge_detection(img)
    cloneImg, contours, heriachy = cv.findContours(binary,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    print(contours[0].shape) # (6,1,2) 6个(x,y)信息
    # contours n个轮廓信息 轮廓信息为numpy数组
    for i, contour in enumerate(contours):
        cv.drawContours(img,contours,i,(0,0,255),2)
        # print(i)
    cv.imshow('contours_demo',img)


"""
矩是描述图像特征以及形状的概率分布的算子，
设 X 和 Y 是随机变量，c 为常数，k 为正整数，
如果E(|X−c|^k)存在，则称E(|X−c|^k)为X关于点c的k阶矩。
c = 0 时， 称为 k 阶原点矩；
c = E(x) 时，称为 k 阶中心矩。
如果E(|X-c1|^p·|Y-c2|^q)存在，则称其为x，y关于点c的p+q阶矩
c1=c2=0时，称为p+q阶混合中心矩
c1=E(X), c2=E(Y)时，称为p+q阶混合中心矩
如果X, Y是连续型的，则下式称其为X，Y关于点c的p+q阶矩
零阶矩:M00可以用来求二值图像（轮廓，连通域）的面积。
一阶矩:M10 就是图像上所以白色区域 x坐标值的累加。因此，一阶矩可以用来求二值图像的重心:
二阶矩:二阶矩可以用来求物体形状的方向。


Douglas-Peucker算法
起始曲线是一组有序的点或线，距离参数ε>0。该算法递归地划分该行。给出起点和终点的所有点，保留第一个和最后一个点，连线。
然后找到距离线段最远的点，如果该点到线的距离小于ε，则可以丢弃当前未标记的任何点。如果距离线段最远的点距离近似值大于ε，
则保持该点。取消当前起点终点的线，同时该点连线到起点终点。依次递归，知道只到点到直线到距离都小于ε。



轮廓的面积可以使用函数 cv2.contourArea()计算得到，也可以使用矩(0阶矩),M['m00']。
cv.contourArea(contour[, oriented]) -> retval

轮廓的弧长 可以用来指定对象的形状是闭合的（True），
cv.arcLength(curve, closed) -> retval

cv.moments(array[, binaryImage]) -> retval
返回值为 计算得到的矩以一个字典的形式。

轮廓近似
approxPolyDP(curve, epsilon, closed[, approxCurve]) -> approxCurve
epsilon是从原始轮廓到近似轮廓的最大距离。它是一个准确度参数。选择一个好的 epsilon 对于得到满意结果非常重要。

边界矩形 一个直矩形（就是没有旋转的矩形）。它不会考虑对象是否旋转。
cv.boundingRect(points) -> retval
返回值为x,y,w,h,    (x，y)为矩形左上角的坐标,(w，h)是矩形的宽和高。


"""

def measure_demo(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255,cv.THRESH_BINARY|cv.THRESH_OTSU)
    print ('threshold is :', ret)
    cv.imshow('binary', binary)

    outImg, contours, hireachy = cv.findContours(binary,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    for i,contour in enumerate(contours):
        area = cv.contourArea(contour)
        x,y,w,h = cv.boundingRect(contour)
        rate = min(w, h)/max(w, h) # 横纵比
        mm = cv.moments(contour)
        type(mm)
        cx = mm['m10']/mm['m00']
        cy = mm['m01']/mm['m00']
        cv.circle(img, (np.int(cx),np.int(cy)), 3, (0,255,255), -1) # 用黄色原点把重心表示出来
        # cv.rectangle(image, (x,y), (x+w, y+h), (0,0,255), 2)
        approxCurve = cv.approxPolyDP(contour, 4, True)
        print(approxCurve.shape)  # (n, 1, 2)  n边行  也就是 n(x,y) 信息
        if approxCurve.shape[0] > 4:
            cv.drawContours(img, contours, i, (0,255,0), 2)

        if approxCurve.shape[0] == 4:
            cv.drawContours(img, contours, i, (255,0,0), 2)

        if approxCurve.shape[0] == 3: # 三角形
            cv.drawContours(img, contours, i, (0,0,255), 2)


        print("contour area %s"%area)
    cv.imshow("measure_contours", img)


"""腐蚀  吞噬1的区域
这个操作会把前景物体的边界腐蚀掉(但是前景仍然是白色)。卷积核沿着图像滑动，如果与卷积核对应的原图像的所有像素值都是1。
那么中心元素就保持原来的像素值，否则就变为零。根据卷积核的大小靠近前景的所有像素都会被腐蚀掉(变为0)。
所以前景物体会变小，整幅图像的白色区域会减少。这对于去除白噪声很有用，也可以用来断开两个连在一块的物体等。

getStructuringElement(shape, ksize[, anchor]) -> retval
    这个函数的第一个参数表示内核的形状，有三种形状可以选择。
        矩形：MORPH_RECT;
        交叉形：MORPH_CORSS;   十字形状
        椭圆形：MORPH_ELLIPSE;
    第二和第三个参数分别是内核的尺寸以及锚点的位置。对于锚点的位置，有默认值Point()-1,-1)表示锚点位于中心点。
    element形状唯一依赖锚点位置，其他情况下，锚点只是影响了形态学运算结果的偏移。


cv.erode(src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]) -> dst

"""
def erode_demo (img):
    print(img.shape)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow('binary', binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))  # 使用np.ones方便
    dst = cv.erode(binary, kernel)
    cv.imshow('erode', dst)


"""膨胀 吞噬0的区域
与卷积核对应的原图像的像素值中只要有一个是1，中心元素的像素值就是1。所以这个操作会增加图像中的白色区域(前景)。膨胀也可以用来连接两个分开的物体。
一般在去噪声时先用腐蚀再用膨胀。因为腐蚀在去掉白噪声的同时，也会使前景对象变小。所以我们再对他进行膨胀。
这时噪声已经被去除了，不会再回来了，但是前景还在并会增加。
cv.dilate(src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]) -> dst

"""

def dilate_demo(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)

    cv.imshow('binary',binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
    dst = cv.dilate(binary,kernel)
    cv.imshow('dilate', dst)

"""开运算  先腐蚀再膨胀

一般在去噪声时先用腐蚀再用膨胀。因为腐蚀在去掉白噪声的同时，也会使前景对象变小。所以我们再对他进行膨胀。
这时噪声已经被去除了，不会再回来了，但是前景还在并会增加。
cv.morphologyEx(src, op, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]) -> dst
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

"""

def open_demo(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)
    cv.imshow('binary', binary)

    kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
    dst = cv.morphologyEx(binary,cv.MORPH_OPEN,kernel)

    cv.imshow('open',dst)




"""闭运算  先膨胀再腐蚀

经常被用来填充前景物体中的小洞，或者前景物体上的小黑点。
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

"""

def close_demo(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)
    cv.imshow('binary',binary)

    kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
    dst = cv.morphologyEx (binary,cv.MORPH_CLOSE,kernel)
    cv.imshow ('close',dst)

"""形态学梯度 膨胀-腐蚀

其实就是一幅图像膨胀与腐蚀的差别。 结果看上去就像前景物体的轮廓。
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

"""
def gradient_demo(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
    dst = cv.morphologyEx(binary, cv.MORPH_GRADIENT, kernel)
    cv.imshow('gradient_demo', dst)

"""形态学梯度  原图-腐蚀   膨胀-原图
基本梯度 膨胀后图像 - 腐蚀后图像
内部梯度 原图像 - 腐蚀后图像
基本梯度 膨胀后图像 - 原图像
"""
def gradient_inner_demo(img):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
    gi = cv.erode(img, kernel)
    go = cv.dilate(img, kernel)
    dst1 = cv.subtract(img, gi)
    dst2 = cv.subtract(go, img)
    cv.imshow('inner', dst1)
    cv.imshow('outter', dst2)

""" 礼帽
原始图像与进行开运算之后得到的图像的差。
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
"""
def tophat_demo(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
    dst = cv.morphologyEx(gray, cv.MORPH_TOPHAT, kernel)
    cimg = np.array(gray.shape, np.uint8)
    cimg = 200
    cv.add(dst, cimg)
    cv.imshow("tophat_demo", dst)

def tophat_binary_demo(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
    dst = cv.morphologyEx(binary, cv.MORPH_TOPHAT, kernel)
    cv.imshow("tophat_demo2", dst)

"""  黑帽
进行闭运算之后得到的图像与原始图像的差。
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
"""
def blackhat_demo(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (25,25))
    dst = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, kernel)
    cimg = np.array(gray.shape, np.uint8)
    cimg = 100
    cv.add(dst, cimg)
    cv.imshow("blackhat_demo", dst)

def blackhat_binary_demo(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (25,25))
    dst = cv.morphologyEx(binary, cv.MORPH_BLACKHAT, kernel)
    cv.imshow("blackhat_demo2", dst)

src = cv.imread('image0.jpg',1)
src1 = cv.imread('image_changed.jpg',1)
src2 =cv.imread('test1.jpeg',1)
src3 = cv.imread('star.png',1)
src_j = cv.imread('j.png',1)
# cv.namedWindow('lpls',cv.WINDOW_NORMAL)
t1 = cv.getTickCount()

# sober_demo(src)
# laplacian_demo(src)
# canny_demo(src2)
# hough_detection_demo(src2)
# hough_detection_possible_demo(src2)
# hough_detection_circles_demo(src)
# contours_demo(src)
# measure_demo(src3)
# erode_demo(src_j)
# dilate_demo(src_j)
# open_demo(src_j)
# close_demo(src_j)
# gradient_demo(src_j)
# gradient_inner_demo(src_j)
# tophat_demo(src)
# tophat_binary_demo(src)
# blackhat_demo(src)
# blackhat_binary_demo(src)


t2 = cv.getTickCount()
time = (t2-t1)/cv.getTickFrequency()
print('time : %s s'%(time))
cv.waitKey(0)
