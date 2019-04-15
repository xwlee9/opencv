import cv2 as cv
import numpy as np

img = cv.imread('image0.jpg',1)  #(filename, flag) flag 彩色3通道读取 默认为1.若为0则灰度返回 若为1则原图返回
cv.namedWindow("image",cv.WINDOW_NORMAL)
"""
WINDOW_NORMAL设置了这个值，用户便可以改变窗口的大小（没有限制)
INDOW_AUTOSIZE如果设置了这个值，窗口大小会自动调整以适应所显示的图像，并且不能手动改变窗口大小.
WINDOW_OPENGL 如果设置了这个值的话，窗口创建的时候便会支持OpenGL
"""

(b,g,r) = img[100,100]  #行对应图像的高 列对应图像的宽
print(b,g,r)            #39 46 49


print(type(img))        #<class 'numpy.ndarray'>
print(img.shape)        # (547, 730, 3)
print(img.size)         # 1197930       547*730*3
print(img.dtype)        # uint8         2**8 = 256
pixel_data = np.array(img)
print(pixel_data)
print(img)


cv.imshow('image',img)   # (namedWindow,data)

# 图片修改
for i in range(0,100):
    img[10+i, 100] = (255,0,0)   # 在第10-109像素位置位置 改成蓝色
cv.imshow('image_changed',img)

#图片压缩
cv.imwrite('image_changed.jpg', img, [cv.IMWRITE_JPEG_QUALITY,50]) # 1 name 2 data
cv.imwrite('image_changed.png', img, [cv.IMWRITE_PNG_COMPRESSION,0])
# jpg 0 压缩比高0-100 有损压缩 png 0 压缩比低0-9

cv.waitKey(0)
cv.destroyAllWindows()
