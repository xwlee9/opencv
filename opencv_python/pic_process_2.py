import cv2 as cv
import numpy as np

"""图片的缩放
最近临域插值 双线性插值 原理
src 10*20 dst 5*10
dst<-src
(1,2) <- (2,4)
dst x 1 -> src x 2 newX
newX = x*(src 行/目标 行) newX = 1*（10/5） = 2
newY = y*(src 列/目标 列) newY = 2*（20/10）= 4
12.3 = 12

双线性插值
A1 = 20% 上+80%下 A2
B1 = 30% 左+70%右 B2
1 最终点  = A1 30% + A2 70%
2 最终点  = B1 20% + B2 80%
"""
def pic_scale_api(img):
    imgInfo = img.shape
    # print(imgInfo)              #(547, 730, 3)
    height = imgInfo[0]
    width = imgInfo[1]
    mode = imgInfo[2]

    dstHeight = int(height*0.5)
    dstWidth = int(width*0.5)

    dst = cv.resize(img,(dstWidth, dstHeight))
    cv.imshow('image', dst)

def pic_scale(img):
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]

    dstHeight = int(height/2)
    dstWidth = int(width/2)

    dstImage = np.zeros((dstHeight,dstWidth,3),np.uint8)#0-255
    for i in range(0,dstHeight):#行
        for j in range(0,dstWidth):#列
            iN = int(i*(height*1.0/dstHeight))
            jN = int(j*(width*1.0/dstWidth))
            dstImage[i,j] = img[iN,jN]
    cv.imshow('image',dstImage)



def pic_scale_big(img):
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    matScale = np.float32([[2,0,0],[0,2,0]])
    dst = cv.warpAffine(img,matScale,(int(width*2),int(height*2)))
    cv.imshow('dst',dst)


"""图片剪切
直接截取
"""
def pic_cut(img):
    imgInfo = img.shape
    dst = img[0:200,100:300]
    cv.imshow('image',dst)

"""图片移位
[1,0,100],[0,1,200] 2*2 2*1
[[1,0],[0,1]]  2*2  A
[[100],[200]] 2*1   B
xy C
A*C+B = [[1*x+0*y],[0*x+1*y]]+[[100],[200]]
= [[x+100],[y+200]]

(10,20)->(110,120)
"""
def pic_shift_api(img):
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]

    matShift = np.float32([[1,0,100],[0,1,200]])# 2*3  [1,0,100] 像素位置x+100, [0,1,200] 像素位置y+200
    dst = cv.warpAffine(img,matShift,(height,width))#1 data 2 mat 3 info
    cv.imshow("image",dst)

def pic_shift(img):
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    dst = np.zeros(img.shape,np.uint8)
    for i in range(0,height):
        for j in range(0,width-200):
            dst[i,j+200]=img[i,j]  # 像素位置 y+200
    cv.imshow("image",dst)


"""图像镜像操作"""
def pic_mirroring(img):
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    deep = imgInfo[2]
    newImgInfo =(height*2, width, deep)
    dst = np.zeros(newImgInfo,np.uint8)
    for i in range(0,height):
        for j in range(0,width):
            dst[i,j]=img[i,j]  #
            dst[2*height-i-1,j] = img[i,j]
    for i in range(0,width):
        dst[height, i] = (0,0,255)
    cv.imshow('image',dst)

""" 仿射变换 """
def pic_affin(img):
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    #src 3->dst 3 (左上角 左下角 右上角)
    matSrc = np.float32([[0,0],[0,height-1],[width-1,0]])
    matDst = np.float32([[50,50],[300,height-200],[width-300,100]])
    #组合
    matAffine = cv.getAffineTransform(matSrc,matDst)# mat 1 src 2 dst
    dst = cv.warpAffine(img,matAffine,(width,height))
    cv.imshow('image', dst)

""" 图片旋转 """
def pic_rotation(img):
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    # 2*3
    matRotate = cv.getRotationMatrix2D((height*0.5,width*0.5),45,0.5)# mat rotate 逆时针旋转1 center 2 angle 3 scale
    #100*100 25
    dst = cv.warpAffine(img,matRotate,(height,width))
    cv.imshow('image', dst)

src = cv.imread('image0.jpg',1)
cv.namedWindow("image",cv.WINDOW_NORMAL)

# pic_scale_api(src)
# pic_scale(src)
# pic_scale_big(src)
# pic_cut(src)
# pic_shift_api(src)
# pic_shift(src)
# pic_mirroring(src)
# pic_affin(src)
pic_rotation(src)


cv.waitKey(0)
cv.destroyAllWindows()
