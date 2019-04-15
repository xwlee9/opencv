import cv2 as cv
import numpy as np

"""简单创建一个图片"""
def create_image():
    img = np.zeros([600,600,3],np.uint8)
    img[:,:,0] = np.ones([600,600])*255 # 将b channel 全部填充255 其他两个channel 填充0
    cv.imshow('image', img)

""" 图片的分割 """

def split_demo(img):
    b, g, r = cv.split(img)
    b1 = img[:,:,0]
    g1 = img[:,:,1]
    r1 = img[:,:,2]
    print(b)
    print(b1)
    cv.imshow('blue', b)
    cv.imshow('blue1', b1)
    cv.imshow('green', g)
    cv.imshow('red', r)
    img[:,:,2] = 0
    cv.imshow('image',img)
    dst = cv.merge([b,g,r])
    dst1 = np.array(img.shape,dtype=np.uint8)
    # dst1 ===> for 循环
    cv.imshow('dst',dst)
    # cv.imshow('dst1',dst1)


""" 颜色反转
255 - pixel
"""
def access_pixels(img):
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    deep = imgInfo[2]
    for row in range(height):
        for col in range(width):
            for de in range(deep):
                pv = img[row,col,de]
                img[row,col,de] = 255-pv
    cv.imshow('pixels_demo',img)

# 调用api
"""图像像素运算
void bitwise_and(InputArray src1, InputArray src2,OutputArray dst, InputArray mask=noArray());#dst = src1 & src2
void bitwise_or(InputArray src1, InputArray src2,OutputArray dst, InputArray mask=noArray());#dst = src1 | src2
void bitwise_xor(InputArray src1, InputArray src2,OutputArray dst, InputArray mask=noArray());#dst = src1 ^ src2
void bitwise_not(InputArray src, OutputArray dst,InputArray mask=noArray());#dst = ~src
"""
def access_pixel_api(img):
    dst = cv.bitwise_not (img,img)  # 按位取反操作
    cv.imshow('inverse demo',dst)


"""色彩空间的转换
灰色 hsv yus Ycrcb
灰色也可以用 cv.imread('image.jpg', 0)
"""
def color_space(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)  # 这个操作不可逆
    cv.imshow('gray', gray)
    hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    cv.imshow('hsv',hsv)
    yuv = cv.cvtColor(img,cv.COLOR_BGR2YUV)
    cv.imshow('yus',yuv)
    Ycrcb = cv.cvtColor(img,cv.COLOR_BGR2YCR_CB)
    cv.imshow('Ycrcb',Ycrcb)

"""取特定颜色
cv2.inRange(hsv, lower_red, upper_red)
第一个参数：hsv指的是原图
第二个参数：lower_red指的是图像中低于这个lower_red的值，图像值变为0
第三个参数：upper_red指的是图像中高于这个upper_red的值，图像值变为0
而在lower_red～upper_red之间的值变成255

掩模运算
用选定的图像、图形或物体，对处理的图像（全部或局部）进行遮挡，来控制图像处理的区域或处理过程。
也就是说 用预先制作的感兴趣区掩模与待处理图像相乘,得到ROI,感兴趣区内图像值保持不变,而区外图像值都为0。
1值区域被处理，被屏蔽的0值区域不被包括在计算中
"""
def extrace_object():
    capture = cv.VideoCapture(0)
    while (True):
        ret,frame = capture.read()
        if ret == False:
            break
        hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
        lower_hsv = np.array([100,43,46])
        upper_hsv = np.array([124,255,255])  # 取蓝色
        mask = cv.inRange(hsv,lowerb = lower_hsv,upperb = upper_hsv)
        dst = cv.bitwise_and(frame,frame,mask=mask)  # 将我们的图像与掩膜运算 也就是只保留hsv的蓝色区域
        cv.imshow('video',frame)
        cv.imshow('mask',dst)
        c = cv.waitKey(25)
        if c == 27:
            break

"""加法运算
一种是消除图像的随机噪声，主要做是讲同一场景的带噪图像的多帧叠加平均降噪；
另一种是用来做特效，把多幅图像叠加在一起，再进一步进行处理。
"""
def add_demo(img1,img2):
    dst = cv.add(img1,img2)
    cv.imshow('dst_demo',dst)

"""减法运算
可以用于目标检测 增强图像间的差别 天文摄影中也需要拍摄暗场,暗平场便于后期处理时减法降噪
"""
def subtract_demo(img1,img2):
    dst = cv.subtract(img1,img2)
    cv.imshow('dst_demo',dst)
"""除法运算
可以用于改变图像的灰度级
"""
def divide_demo(img1,img2):
    dst = cv.divide(img1,img2)
    cv.imshow('dst_demo1',dst)

"""乘法运算
抑制图像的某些区域，掩膜值置为1，否则置为0，提取ROI区域 乘运算有时也被用来实现卷积或相关的运算。
"""
def multiply_demo(img1,img2):
    dst = cv.multiply(img1,img2)
    cv.imshow('dst_demo',dst)

"""逻辑运算
and 与运算 二值中 两图像白色相交为白色
or 或运算  二值中 两图像白色均为白色
xor 异或运算  二值中 两图像白色相交为黑色 其余为白色
"""
def logic_demo(img1,img2):
    dst1 = cv.bitwise_and(img1,img2)
    dst2 = cv.bitwise_or(img1,img2)
    cv.imshow('dst_demo',dst1)
    cv.imshow('dst_demo1',dst2)

"""亮度增强
addWeighted(InputArray src1, double alpha, InputArray src2, double beta, double gamma, OutputArray dst, int dtype=-1)
参数1：src1，第一个原数组.
参数2：alpha，第一个数组元素权重
参数3：src2第二个原数组
参数4：beta，第二个数组元素权重
参数5：gamma，图1与图2作和后添加的数值。
"""
def contrast_brightness_demo(img,c,b):
    h , w , d = img.shape
    blank = np.zeros([h,w,d],img.dtype)
    dst = cv.addWeighted(img,c,blank,1-c,b)  # c > 1 ==> img * c + b
    print(img[1,1])
    print(dst[1,1])

    cv.imshow('con_bri_demo',dst)

"""计算均值方差
cv.meanStdDev()
返回均值和方差
"""
def others_demo(img):
    mean, stddv = cv.meanStdDev(img)
    print(mean)  # 3个channel的均值方差
    print(stddv)

"""洪填充
floodFill函数： 颜色替换
floodFill(InputOutputArray image, Point seedPoint, Scalar newVal, Rect* rect=0,
          Scalar loDiff=Scalar(), Scalar upDiff=Scalar(), int flags=4 ) 
1.操作的图像, 2.掩模, 3.起始像素值，4.填充的颜色, 5.填充颜色的低值， 6.填充颜色的高值 ,7.填充的方法
参数2:只有mask参数中像素值为0的区域才会被填充
参数5:填充颜色的低值就是：参数3像素值 减去 参数5
参数6:填充颜色的高值就是：参数3像素值 加上 参数6  ===> 即是这两个数值之间的色素替换为参数4的颜色

彩色图像一般是FLOODFILL_FIXED_RANGE 指定颜色填充 待处理的像素点与种子点作比较
如果满足(s – lodiff , s + updiff)之间(s为种子点像素值)则填充此像素
另外一种是FLOODFILL_MASK_ONLY，mask的指定的位置为零时才填充，不为零不填充

"""
def fill_color_demo(img):
    img_cp = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros([h+2,w+2],np.uint8) # 该版本特有的掩膜 +2是官方函数要求
    cv.floodFill(img_cp,mask,(60,60),(0,0,255),(100,100,100),(50,50,50),cv.FLOODFILL_FIXED_RANGE)
    cv.imshow('fill_color_demo',img_cp)

def fill_binary_demo():
    img = np.zeros([600,600,3],np.uint8)
    img[200:500,200:500,:] = 255
    cv.imshow('fill_binary_demo',img)
    mask = np.ones([602,602,1],np.uint8)
    mask[201:501,201:501,:] = 0
    cv.floodFill(img,mask,(200,200),(0,0,255),cv.FLOODFILL_FIXED_RANGE)
    cv.imshow('fill_binary',img)

""" 模糊
均值模糊 可以有效的平滑图像，降低图像的尖锐程度，降低噪声。但缺点是不能消除噪声。
中值模糊 它不创造新的像素值 取周围像素值作为它的输出，可以有效的消除脉冲噪声，椒盐噪声 很好的保护图像尖锐的边缘 

 """
def blur_demo(img):
    dst = cv.blur(img,(5,5))  # (src, kernel) 5*5
    cv.imshow('blur_demo',dst)

def median_blur_demo(img):
    dst = cv.medianBlur(img,3)  # (src, ksize) ksize:必须大于1还是奇数
    cv.imshow('medianBlur',dst)

# 避免像素越界
def clasp(pv):
    if pv > 255:
        return 255
    if pv < 0:
        return 0
    else:
        return pv

""" 高斯滤波
高斯滤波 计算平均值的时候，只需要将"中心点"作为原点，其他点按照其在正态曲线上的位置分配权重，得到一个加权平均值。
        平滑像素 消除高斯噪声
  cv.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]]) -> dst
  src和dst分别是输入图像和输出图像 Ksize为高斯滤波器模板大小，sigmaX和sigmaY分别为高斯滤波在横线和竖向的滤波系数即高斯核在xy方向的标准差
"""

# 添加高斯噪声
def guassian_noise(img):
    h,w,d = img.shape
    for row in range(h):
        for col in range(w):
            s = np.random.normal(0,20,3)
            b = img[row,col,0]
            g = img[row,col,1]
            r = img[row,col,2]
            img[row,col,0] = clasp(b + s[0])
            img[row,col,1] = clasp(g + s[1])
            img[row,col,2] = clasp(r + s[2])
    cv.imshow('guassian_noise',img)
    return img
# 高斯滤波
def guassian_blur(img):
    img_n = guassian_noise(img)
    dst = cv.GaussianBlur(img_n,(5,5),1,1)
    cv.imshow('after guassian_blur', dst)

"""

双边滤波 双边滤波是保留边缘的滤波方法，避免了边缘信息的丢失 保留了图像轮廓不变
   void bilateralFilter(InputArray src, OutputArray dst, int d, double sigmaColor,
                        double sigmaSpace, int borderType=BORDER_DEFAULT )
   int d：表示在过滤过程中每个像素邻域的直径范围。如果这个值是非正数，则函数会从第五个参数sigmaSpace计算该值 
   double sigmaColor：色彩空间的sigma参数，该参数较大时，各像素邻域内相距较远的颜色会被混合到一起，从而造成更大范围的半相等颜色
   double sigmaSpace：坐标空间的sigma参数，该参数较大时，只要颜色相近，越远的像素会相互影响同的颜色。
                    当d > 0时，d指定了邻域大小且与sigmaSpace无关，否则d正比于sigmaSpace。
"""
# 双边滤波
def bi_demo(img):
    dst = cv.bilateralFilter(img,0,100,15)
    cv.imshow('dst',dst)

"""滤波器
cv.filter2D(src,ddepth,kernel=kernel)

均值漂移 对于给定的一定数量样本，任选其中一个样本，以该样本为中心点划定一个圆形区域，
        求取该圆形区域内样本的质心，即密度最大处的点，再以该点为中心继续执行上述迭代过程，直至最终收敛。
        图像在色彩层面的平滑滤波，它可以中和色彩分布相近的颜色，平滑色彩细节，侵蚀掉面积较小的颜色区域

cv.pyrMeanShiftFiltering(src, sp, sr[, dst[, maxLevel[, termcrit]]]) -> dst
    参数src，输入图像，8位，三通道的彩色图像，并不要求必须是RGB格式，HSV、YUV等Opencv中的彩色图像格式均可；
    参数sp，定义的漂移物理空间半径大小；
    参数sr，定义的漂移色彩空间半径大小；
    参数maxLevel，定义金字塔的最大层数；
    参数termcrit，定义的漂移迭代终止条件，可以设置为迭代次数满足终止，迭代目标与中心点偏差满足终止，或者两者的结合；
    dst，输出图像，跟输入src有同样的大小和数据格式；
"""

def filter2D_demo(img):
    # kernel = np.ones([5,5],np.float32)/25
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],np.float32)   # 锐化
    dst = cv.filter2D(img,-1,kernel=kernel)
    #int ddepth: 目标图像深度，如果没写或者当ddepth输入值为-1时，将生成与原图像深度相同的图像
    cv.imshow('filter2D',dst)

def pyrMeanShiftFiltering_demo(img):
    dst = cv.pyrMeanShiftFiltering(img,-1,10,10)
    cv.imshow('pyrMeanShiftFiltering',dst)


"""========================================================================="""


src = cv.imread('image0.jpg',1)
src1 = cv.imread('image_changed.jpg',1)
cv.namedWindow("image",cv.WINDOW_NORMAL)


""" 增加对处理时间的计算
getTickCount()：用于返回从操作系统启动到当前所经的计时周期数，看名字也很好理解，get Tick Count(s)。
getTickFrequency()：用于返回CPU的频率。get Tick Frequency。这里的单位是秒，也就是一秒内重复的次数。
"""
t1 = cv.getTickCount()
# create_image()
# split_demo(src)
# access_pixels(src)
# access_pixel_api(src)

# color_space(src)
# extrace_object()

""" 运算 """
# add_demo(src, src1)
# subtract_demo(src, src1)
# divide_demo(src, src1)
# multiply_demo(src, src1)
# logic_demo(src, src1)
# contrast_brightness_demo(src, 1.1, 10)
# others_demo(src)
"""洪填充"""
# fill_color_demo(src)
# fill_binary_demo()
"""模糊"""
# blur_demo(src)
# median_blur_demo(src1)
# guassian_blur(src1)
# bi_demo(src1)

"""滤波器"""
# filter2D_demo(src1)
# pyrMeanShiftFiltering_demo(src1)



t2 = cv.getTickCount()
time = (t2-t1)/cv.getTickFrequency()
print('time : %s s'%(time))

cv.waitKey(0)
cv.destroyAllWindows()
