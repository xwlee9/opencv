#include <opencv2/opencv.hpp>  //头文件
using namespace cv;  //包含cv命名空间

void main( )
{
	// 【1】读入一张图片，载入图像
	Mat srcImage = imread("1.jpg");
	// 【2】显示载入的图片
	imshow("【原始图】",srcImage);
	// 【3】等待任意按键按下
	waitKey(0);
}
*********** 腐蚀 *************
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//-----------------------------------【命名空间声明部分】---------------------------------------
//		描述：包含程序所使用的命名空间
//-----------------------------------------------------------------------------------------------
using namespace cv;

//-----------------------------------【main( )函数】--------------------------------------------
//		描述：控制台应用程序的入口函数，我们的程序从这里开始
//-----------------------------------------------------------------------------------------------
int main(   )
{
	//载入原图
	Mat srcImage = imread("1.jpg");
	//显示原图
	imshow("【原图】腐蚀操作", srcImage);
	//进行腐蚀操作
	Mat element = getStructuringElement(MORPH_RECT, Size(50, 50));
	Mat dstImage;
	erode(srcImage, dstImage, element);
	//显示效果图
	imshow("【效果图】腐蚀操作", dstImage);
	waitKey(0);

	return 0;
}
***************blur 图像模糊***********

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;

//-----------------------------------【main( )函数】--------------------------------------------
//		描述：控制台应用程序的入口函数，我们的程序从这里开始
//-----------------------------------------------------------------------------------------------
int main( )
{
	//【1】载入原始图
	Mat srcImage=imread("1.jpg");

	//【2】显示原始图
	imshow( "均值滤波【原图】", srcImage );

	//【3】进行均值滤波操作
	Mat dstImage;
	blur( srcImage, dstImage, Size(7, 7));

	//【4】显示效果图
	imshow( "均值滤波【效果图】" ,dstImage );

	waitKey( 0 );
}
***********canny边缘检测********

#include <opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
using namespace cv;

//-----------------------------------【main( )函数】--------------------------------------------
//		描述：控制台应用程序的入口函数，我们的程序从这里开始
//-------------------------------------------------------------------------------------------------
int main( )
{
	//【0】载入原始图
	Mat srcImage = imread("1.jpg");  //工程目录下应该有一张名为1.jpg的素材图
	imshow("【原始图】Canny边缘检测", srcImage); 	//显示原始图
	Mat dstImage,edge,grayImage;	//参数定义

	//【1】创建与src同类型和大小的矩阵(dst)
	dstImage.create( srcImage.size(), srcImage.type() );

	//【2】将原图像转换为灰度图像
	//此句代码的OpenCV2版为：
	//cvtColor( srcImage, grayImage, CV_BGR2GRAY );
	//此句代码的OpenCV3版为：
	cvtColor( srcImage, grayImage, COLOR_BGR2GRAY );

	//【3】先用使用 3x3内核来降噪
	blur( grayImage, edge, Size(3,3) );

	//【4】运行Canny算子
	Canny( edge, edge, 3, 9,3 );

	//【5】显示效果图
	imshow("【效果图】Canny边缘检测", edge);

	waitKey(0);

	return 0;
}
************播放视频********** 把1.avi换成0就是摄像头采集视频

#include <opencv2/opencv.hpp>
using namespace cv;

//-----------------------------------【main( )函数】--------------------------------------------
//		描述：控制台应用程序的入口函数，我们的程序从这里开始
//-------------------------------------------------------------------------------------------------
int main( )
{
	//【1】读入视频
	VideoCapture capture("1.avi");

	//【2】循环显示每一帧
	while(1)
	{
		Mat frame;//定义一个Mat变量，用于存储每一帧的图像
		capture>>frame;  //读取当前帧
		imshow("读取视频",frame);  //显示当前帧
		waitKey(30);  //延时30ms
	}
	return 0;
}
*********图像载入与输出**********
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;


int main( )
{
	//-----------------------------------【一、图像的载入和显示】---------------------------------
	//	描述：以下三行代码用于完成图像的载入和显示
	//--------------------------------------------------------------------------------------------------

	Mat girl=imread("girl.jpg"); //载入图像到Mat
	namedWindow("【1】动漫图"); //创建一个名为 "【1】动漫图"的窗口
	imshow("【1】动漫图",girl);//显示名为 "【1】动漫图"的窗口

	//-----------------------------------【二、初级图像混合】--------------------------------------
	//	描述：二、初级图像混合
	//--------------------------------------------------------------------------------------------------
	//载入图片
	Mat image= imread("dota.jpg",199);
	Mat logo= imread("dota_logo.jpg");

	//载入后先显示
	namedWindow("【2】原画图");
	imshow("【2】原画图",image);

	namedWindow("【3】logo图");
	imshow("【3】logo图",logo);

	// 定义一个Mat类型，用于存放，图像的ROI
	Mat imageROI;
	//方法一
	imageROI= image(Rect(800,350,logo.cols,logo.rows));
	//方法二
	//imageROI= image(Range(350,350+logo.rows),Range(800,800+logo.cols));

	// 将logo加到原图上
	addWeighted(imageROI,0.5,logo,0.3,0.,imageROI);

	//显示结果
	namedWindow("【4】原画+logo图");
	imshow("【4】原画+logo图",image);

	//-----------------------------------【三、图像的输出】--------------------------------------
	//	描述：将一个Mat图像输出到图像文件
	//-----------------------------------------------------------------------------------------------
	//输出一张jpg图片到工程目录下
	imwrite("由imwrite生成的图片.jpg",image);

	waitKey();

	return 0;
}

************ 滑动条 **********

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

//-----------------------------------【宏定义部分】--------------------------------------------
//  描述：定义一些辅助宏
//------------------------------------------------------------------------------------------------
#define WINDOW_NAME "【滑动条的创建&线性混合示例】"        //为窗口标题定义的宏


//-----------------------------------【全局变量声明部分】--------------------------------------
//		描述：全局变量声明
//-----------------------------------------------------------------------------------------------
const int g_nMaxAlphaValue = 100;//Alpha值的最大值
int g_nAlphaValueSlider;//滑动条对应的变量
double g_dAlphaValue;
double g_dBetaValue;

//声明存储图像的变量
Mat g_srcImage1;
Mat g_srcImage2;
Mat g_dstImage;


//-----------------------------------【on_Trackbar( )函数】--------------------------------
//		描述：响应滑动条的回调函数
//------------------------------------------------------------------------------------------
void on_Trackbar( int, void* )
{
	//求出当前alpha值相对于最大值的比例
	g_dAlphaValue = (double) g_nAlphaValueSlider/g_nMaxAlphaValue ;
	//则beta值为1减去alpha值
	g_dBetaValue = ( 1.0 - g_dAlphaValue );

	//根据alpha和beta值进行线性混合
	addWeighted( g_srcImage1, g_dAlphaValue, g_srcImage2, g_dBetaValue, 0.0, g_dstImage);

	//显示效果图
	imshow( WINDOW_NAME, g_dstImage );
}


//-----------------------------【ShowHelpText( )函数】--------------------------------------
//		描述：输出帮助信息
//-------------------------------------------------------------------------------------------------
//-----------------------------------【ShowHelpText( )函数】----------------------------------
//		描述：输出一些帮助信息
//----------------------------------------------------------------------------------------------
void ShowHelpText()
{
	//输出欢迎信息和OpenCV版本
	printf("\n\n\t\t\t非常感谢购买《OpenCV3编程入门》一书！\n");
	printf("\n\n\t\t\t此为本书OpenCV3版的第17个配套示例程序\n");
	printf("\n\n\t\t\t   当前使用的OpenCV版本为：" CV_VERSION );
	printf("\n\n  ----------------------------------------------------------------------------\n");
}


//--------------------------------------【main( )函数】-----------------------------------------
//		描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-----------------------------------------------------------------------------------------------
int main( int argc, char** argv )
{

	//显示帮助信息
	ShowHelpText();

	//加载图像 (两图像的尺寸需相同)
	g_srcImage1 = imread("1.jpg");
	g_srcImage2 = imread("2.jpg");
	if( !g_srcImage1.data ) { printf("读取第一幅图片错误，请确定目录下是否有imread函数指定图片存在~！ \n"); return -1; }
	if( !g_srcImage2.data ) { printf("读取第二幅图片错误，请确定目录下是否有imread函数指定图片存在~！\n"); return -1; }

	//设置滑动条初值为70
	g_nAlphaValueSlider = 70;

	//创建窗体
	namedWindow(WINDOW_NAME, 1);

	//在创建的窗体中创建一个滑动条控件
	char TrackbarName[50];
	sprintf( TrackbarName, "透明值 %d", g_nMaxAlphaValue );

	createTrackbar( TrackbarName, WINDOW_NAME, &g_nAlphaValueSlider, g_nMaxAlphaValue, on_Trackbar );

	//结果在回调函数中显示
	on_Trackbar( g_nAlphaValueSlider, 0 );

	//按任意键退出
	waitKey(0);

	return 0;
}

******** 鼠标输入***********

#include <opencv2/opencv.hpp>
using namespace cv;

//-----------------------------------【宏定义部分】--------------------------------------------
//  描述：定义一些辅助宏
//------------------------------------------------------------------------------------------------
#define WINDOW_NAME "【程序窗口】"        //为窗口标题定义的宏


//-----------------------------------【全局函数声明部分】------------------------------------
//		描述：全局函数的声明
//------------------------------------------------------------------------------------------------
void on_MouseHandle(int event, int x, int y, int flags, void* param);
void DrawRectangle( cv::Mat& img, cv::Rect box );
void ShowHelpText( );

//-----------------------------------【全局变量声明部分】-----------------------------------
//		描述：全局变量的声明
//-----------------------------------------------------------------------------------------------
Rect g_rectangle;
bool g_bDrawingBox = false;//是否进行绘制
RNG g_rng(12345);



//-----------------------------------【main( )函数】--------------------------------------------
//		描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-------------------------------------------------------------------------------------------------
int main( int argc, char** argv )
{
	//【0】改变console字体颜色
	system("color 9F");

	//【0】显示欢迎和帮助文字
	ShowHelpText();

	//【1】准备参数
	g_rectangle = Rect(-1,-1,0,0);
	Mat srcImage(600, 800,CV_8UC3), tempImage;
	srcImage.copyTo(tempImage);
	g_rectangle = Rect(-1,-1,0,0);
	srcImage = Scalar::all(0);

	//【2】设置鼠标操作回调函数
	namedWindow( WINDOW_NAME );
	setMouseCallback(WINDOW_NAME,on_MouseHandle,(void*)&srcImage);

	//【3】程序主循环，当进行绘制的标识符为真时，进行绘制
	while(1)
	{
		srcImage.copyTo(tempImage);//拷贝源图到临时变量
		if( g_bDrawingBox ) DrawRectangle( tempImage, g_rectangle );//当进行绘制的标识符为真，则进行绘制
		imshow( WINDOW_NAME, tempImage );
		if( waitKey( 10 ) == 27 ) break;//按下ESC键，程序退出
	}
	return 0;
}



//--------------------------------【on_MouseHandle( )函数】-----------------------------
//		描述：鼠标回调函数，根据不同的鼠标事件进行不同的操作
//-----------------------------------------------------------------------------------------------
void on_MouseHandle(int event, int x, int y, int flags, void* param)
{

	Mat& image = *(cv::Mat*) param;
	switch( event)
	{
		//鼠标移动消息
	case EVENT_MOUSEMOVE:
		{
			if( g_bDrawingBox )//如果是否进行绘制的标识符为真，则记录下长和宽到RECT型变量中
			{
				g_rectangle.width = x-g_rectangle.x;
				g_rectangle.height = y-g_rectangle.y;
			}
		}
		break;

		//左键按下消息
	case EVENT_LBUTTONDOWN:
		{
			g_bDrawingBox = true;
			g_rectangle =Rect( x, y, 0, 0 );//记录起始点
		}
		break;

		//左键抬起消息
	case EVENT_LBUTTONUP:
		{
			g_bDrawingBox = false;//置标识符为false
			//对宽和高小于0的处理
			if( g_rectangle.width < 0 )
			{
				g_rectangle.x += g_rectangle.width;
				g_rectangle.width *= -1;
			}

			if( g_rectangle.height < 0 )
			{
				g_rectangle.y += g_rectangle.height;
				g_rectangle.height *= -1;
			}
			//调用函数进行绘制
			DrawRectangle( image, g_rectangle );
		}
		break;

	}
}



//-----------------------------------【DrawRectangle( )函数】------------------------------
//		描述：自定义的矩形绘制函数
//-----------------------------------------------------------------------------------------------
void DrawRectangle( cv::Mat& img, cv::Rect box )
{
	cv::rectangle(img,box.tl(),box.br(),cv::Scalar(g_rng.uniform(0, 255), g_rng.uniform(0,255), g_rng.uniform(0,255)));//随机颜色
}


//-----------------------------------【ShowHelpText( )函数】-----------------------------
//          描述：输出一些帮助信息
//----------------------------------------------------------------------------------------------
void ShowHelpText()
{
	//输出欢迎信息和OpenCV版本
	printf("\n\n\t\t\t非常感谢购买《OpenCV3编程入门》一书！\n");
	printf("\n\n\t\t\t此为本书OpenCV3版的第18个配套示例程序\n");
	printf("\n\n\t\t\t   当前使用的OpenCV版本为：" CV_VERSION );
	printf("\n\n  ----------------------------------------------------------------------------\n");
	//输出一些帮助信息
	printf("\n\n\n\t欢迎来到【鼠标交互演示】示例程序\n");
	printf("\n\n\t请在窗口中点击鼠标左键并拖动以绘制矩形\n");

}

**************** 膨胀/腐蚀函数************

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

//-----------------------------------【命名空间声明部分】---------------------------------------
//	描述：包含程序所使用的命名空间
//-----------------------------------------------------------------------------------------------
using namespace std;
using namespace cv;

//-----------------------------------【main( )函数】--------------------------------------------
//	描述：控制台应用程序的入口函数，我们的程序从这里开始
//-----------------------------------------------------------------------------------------------
int main(   )
{

	//载入原图
	Mat image = imread("1.jpg");

	//创建窗口
	namedWindow("【原图】膨胀操作");
	namedWindow("【效果图】膨胀操作");

	//显示原图
	imshow("【原图】膨胀操作", image);

	//进行膨胀操作
	Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
	Mat out;
	dilate(image, out, element);

	//显示效果图
	imshow("【效果图】膨胀操作", out);

	waitKey(0);

	return 0;
}

                        &&&&&&


#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

//-----------------------------------【main( )函数】--------------------------------------------
//	描述：控制台应用程序的入口函数，我们的程序从这里开始
//-----------------------------------------------------------------------------------------------
int main(   )
{
	//载入原图
	Mat srcImage = imread("1.jpg");
	//显示原图
	imshow("【原图】腐蚀操作", srcImage);
	//进行腐蚀操作
	Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
	Mat dstImage;
	erode(srcImage, dstImage, element);
	//显示效果图
	imshow("【效果图】腐蚀操作", dstImage);
	waitKey(0);

	return 0;
}

******************腐蚀与膨胀加轨迹条**************
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
using namespace std;
using namespace cv;


//-----------------------------------【全局变量声明部分】--------------------------------------
//		描述：全局变量声明
//-----------------------------------------------------------------------------------------------
Mat g_srcImage, g_dstImage;//原始图和效果图
int g_nTrackbarNumer = 0;//0表示腐蚀erode, 1表示膨胀dilate
int g_nStructElementSize = 3; //结构元素(内核矩阵)的尺寸


//-----------------------------------【全局函数声明部分】--------------------------------------
//		描述：全局函数声明
//-----------------------------------------------------------------------------------------------
void Process();//膨胀和腐蚀的处理函数
void on_TrackbarNumChange(int, void *);//回调函数
void on_ElementSizeChange(int, void *);//回调函数
void ShowHelpText();

//-----------------------------------【main( )函数】--------------------------------------------
//		描述：控制台应用程序的入口函数，我们的程序从这里开始
//-----------------------------------------------------------------------------------------------
int main( )
{
	//改变console字体颜色
	system("color 2F");

	//载入原图
	g_srcImage = imread("1.jpg");
	if( !g_srcImage.data ) { printf("读取srcImage错误~！ \n"); return false; }

	ShowHelpText();

	//显示原始图
	namedWindow("【原始图】");
	imshow("【原始图】", g_srcImage);

	//进行初次腐蚀操作并显示效果图
	namedWindow("【效果图】");
	//获取自定义核
	Mat element = getStructuringElement(MORPH_RECT, Size(2*g_nStructElementSize+1, 2*g_nStructElementSize+1),Point( g_nStructElementSize, g_nStructElementSize ));
	erode(g_srcImage, g_dstImage, element);
	imshow("【效果图】", g_dstImage);

	//创建轨迹条
	createTrackbar("腐蚀/膨胀", "【效果图】", &g_nTrackbarNumer, 1, on_TrackbarNumChange);
	createTrackbar("内核尺寸", "【效果图】", &g_nStructElementSize, 21, on_ElementSizeChange);

	//输出一些帮助信息
	cout<<endl<<"\t运行成功，请调整滚动条观察图像效果~\n\n"
		<<"\t按下“q”键时，程序退出。\n";

	//轮询获取按键信息，若下q键，程序退出
	while(char(waitKey(1)) != 'q') {}

	return 0;
}

//-----------------------------【Process( )函数】------------------------------------
//		描述：进行自定义的腐蚀和膨胀操作
//-----------------------------------------------------------------------------------------
void Process()
{
	//获取自定义核
	Mat element = getStructuringElement(MORPH_RECT, Size(2*g_nStructElementSize+1, 2*g_nStructElementSize+1),Point( g_nStructElementSize, g_nStructElementSize ));

	//进行腐蚀或膨胀操作
	if(g_nTrackbarNumer == 0) {
		erode(g_srcImage, g_dstImage, element);
	}
	else {
		dilate(g_srcImage, g_dstImage, element);
	}

	//显示效果图
	imshow("【效果图】", g_dstImage);
}


//-----------------------------【on_TrackbarNumChange( )函数】------------------------------------
//		描述：腐蚀和膨胀之间切换开关的回调函数
//-----------------------------------------------------------------------------------------------------
void on_TrackbarNumChange(int, void *)
{
	//腐蚀和膨胀之间效果已经切换，回调函数体内需调用一次Process函数，使改变后的效果立即生效并显示出来
	Process();
}


//-----------------------------【on_ElementSizeChange( )函数】-------------------------------------
//		描述：腐蚀和膨胀操作内核改变时的回调函数
//-----------------------------------------------------------------------------------------------------
void on_ElementSizeChange(int, void *)
{
	//内核尺寸已改变，回调函数体内需调用一次Process函数，使改变后的效果立即生效并显示出来
	Process();
}


//-----------------------------------【ShowHelpText( )函数】-----------------------------
//		 描述：输出一些帮助信息
//----------------------------------------------------------------------------------------------
void ShowHelpText()
{
	//输出欢迎信息和OpenCV版本
	printf("\n\n\t\t\t非常感谢购买《OpenCV3编程入门》一书！\n");
	printf("\n\n\t\t\t此为本书OpenCV3版的第40个配套示例程序\n");
	printf("\n\n\t\t\t   当前使用的OpenCV版本为：" CV_VERSION );
	printf("\n\n  ----------------------------------------------------------------------------\n");
}

*****************形态学滤波************
开运算：（先腐蚀后膨胀）用来消除小物体，在纤细点处分离物体
闭运算：（先膨胀后腐蚀）能排除小型黑洞
形态学梯度：(膨胀图与腐蚀图之差)用来保留物体的边缘轮廓
顶帽：原图像与开运算的结果图之差 用来分离比邻近点亮一些的斑块
黒帽：闭运算的结果图与原图像之差  黒帽运算后的效果图突出了比原图轮廓周围的区域更暗的区域

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;


//-----------------------------------【全局变量声明部分】-----------------------------------
//		描述：全局变量声明
//-----------------------------------------------------------------------------------------------
Mat g_srcImage, g_dstImage;//原始图和效果图
int g_nElementShape = MORPH_RECT;//元素结构的形状

//变量接收的TrackBar位置参数
int g_nMaxIterationNum = 10;
int g_nOpenCloseNum = 0;
int g_nErodeDilateNum = 0;
int g_nTopBlackHatNum = 0;



//-----------------------------------【全局函数声明部分】--------------------------------------
//		描述：全局函数声明
//-----------------------------------------------------------------------------------------------
static void on_OpenClose(int, void*);//回调函数
static void on_ErodeDilate(int, void*);//回调函数
static void on_TopBlackHat(int, void*);//回调函数
static void ShowHelpText();


//-----------------------------------【main( )函数】--------------------------------------------
//		描述：控制台应用程序的入口函数，我们的程序从这里开始
//-----------------------------------------------------------------------------------------------
int main( )
{
	//改变console字体颜色
	system("color 2F");

	ShowHelpText();

	//载入原图
	g_srcImage = imread("1.jpg");
	if( !g_srcImage.data ) { printf("Oh，no，读取srcImage错误~！ \n"); return false; }

	//显示原始图
	namedWindow("【原始图】");
	imshow("【原始图】", g_srcImage);

	//创建三个窗口
	namedWindow("【开运算/闭运算】",1);
	namedWindow("【腐蚀/膨胀】",1);
	namedWindow("【顶帽/黑帽】",1);

	//参数赋值
	g_nOpenCloseNum=9;
	g_nErodeDilateNum=9;
	g_nTopBlackHatNum=2;

	//分别为三个窗口创建滚动条
	createTrackbar("迭代值", "【开运算/闭运算】",&g_nOpenCloseNum,g_nMaxIterationNum*2+1,on_OpenClose);
	createTrackbar("迭代值", "【腐蚀/膨胀】",&g_nErodeDilateNum,g_nMaxIterationNum*2+1,on_ErodeDilate);
	createTrackbar("迭代值", "【顶帽/黑帽】",&g_nTopBlackHatNum,g_nMaxIterationNum*2+1,on_TopBlackHat);

	//轮询获取按键信息
	while(1)
	{
		int c;

		//执行回调函数
		on_OpenClose(g_nOpenCloseNum, 0);
		on_ErodeDilate(g_nErodeDilateNum, 0);
		on_TopBlackHat(g_nTopBlackHatNum,0);

		//获取按键
		c = waitKey(0);

		//按下键盘按键Q或者ESC，程序退出
		if( (char)c == 'q'||(char)c == 27 )
			break;
		//按下键盘按键1，使用椭圆(Elliptic)结构元素结构元素MORPH_ELLIPSE
		if( (char)c == 49 )//键盘按键1的ASII码为49
			g_nElementShape = MORPH_ELLIPSE;
		//按下键盘按键2，使用矩形(Rectangle)结构元素MORPH_RECT
		else if( (char)c == 50 )//键盘按键2的ASII码为50
			g_nElementShape = MORPH_RECT;
		//按下键盘按键3，使用十字形(Cross-shaped)结构元素MORPH_CROSS
		else if( (char)c == 51 )//键盘按键3的ASII码为51
			g_nElementShape = MORPH_CROSS;
		//按下键盘按键space，在矩形、椭圆、十字形结构元素中循环
		else if( (char)c == ' ' )
			g_nElementShape = (g_nElementShape + 1) % 3;
	}

	return 0;
}


//-----------------------------------【on_OpenClose( )函数】----------------------------------
//		描述：【开运算/闭运算】窗口的回调函数
//-----------------------------------------------------------------------------------------------
static void on_OpenClose(int, void*)
{
	//偏移量的定义
	int offset = g_nOpenCloseNum - g_nMaxIterationNum;//偏移量
	int Absolute_offset = offset > 0 ? offset : -offset;//偏移量绝对值
	//自定义核
	Mat element = getStructuringElement(g_nElementShape, Size(Absolute_offset*2+1, Absolute_offset*2+1), Point(Absolute_offset, Absolute_offset) );
	//进行操作
	if( offset < 0 )
		//此句代码的OpenCV2版为：
		//morphologyEx(g_srcImage, g_dstImage, CV_MOP_OPEN, element);
		//此句代码的OpenCV3版为:
		morphologyEx(g_srcImage, g_dstImage, MORPH_OPEN, element);
	else
		//此句代码的OpenCV2版为：
		//morphologyEx(g_srcImage, g_dstImage, CV_MOP_CLOSE, element);
		//此句代码的OpenCV3版为:
		morphologyEx(g_srcImage, g_dstImage, MORPH_CLOSE, element);



	//显示图像
	imshow("【开运算/闭运算】",g_dstImage);
}


//-----------------------------------【on_ErodeDilate( )函数】----------------------------------
//		描述：【腐蚀/膨胀】窗口的回调函数
//-----------------------------------------------------------------------------------------------
static void on_ErodeDilate(int, void*)
{
	//偏移量的定义
	int offset = g_nErodeDilateNum - g_nMaxIterationNum;	//偏移量
	int Absolute_offset = offset > 0 ? offset : -offset;//偏移量绝对值
	//自定义核
	Mat element = getStructuringElement(g_nElementShape, Size(Absolute_offset*2+1, Absolute_offset*2+1), Point(Absolute_offset, Absolute_offset) );
	//进行操作
	if( offset < 0 )
		erode(g_srcImage, g_dstImage, element);
	else
		dilate(g_srcImage, g_dstImage, element);
	//显示图像
	imshow("【腐蚀/膨胀】",g_dstImage);
}


//-----------------------------------【on_TopBlackHat( )函数】--------------------------------
//		描述：【顶帽运算/黑帽运算】窗口的回调函数
//----------------------------------------------------------------------------------------------
static void on_TopBlackHat(int, void*)
{
	//偏移量的定义
	int offset = g_nTopBlackHatNum - g_nMaxIterationNum;//偏移量
	int Absolute_offset = offset > 0 ? offset : -offset;//偏移量绝对值
	//自定义核
	Mat element = getStructuringElement(g_nElementShape, Size(Absolute_offset*2+1, Absolute_offset*2+1), Point(Absolute_offset, Absolute_offset) );
	//进行操作
	if( offset < 0 )
		morphologyEx(g_srcImage, g_dstImage, MORPH_TOPHAT , element);
	else
		morphologyEx(g_srcImage, g_dstImage, MORPH_BLACKHAT, element);
	//显示图像
	imshow("【顶帽/黑帽】",g_dstImage);
}

//-----------------------------------【ShowHelpText( )函数】----------------------------------
//		描述：输出一些帮助信息
//----------------------------------------------------------------------------------------------
static void ShowHelpText()
{
	//输出欢迎信息和OpenCV版本
	printf("\n\n\t\t\t非常感谢购买《OpenCV3编程入门》一书！\n");
	printf("\n\n\t\t\t此为本书OpenCV3版的第48个配套示例程序\n");
	printf("\n\n\t\t\t   当前使用的OpenCV版本为：" CV_VERSION );
	printf("\n\n  ----------------------------------------------------------------------------\n");

	//输出一些帮助信息
	printf("\n\t请调整滚动条观察图像效果\n\n");
	printf( "\n\t按键操作说明: \n\n"
		"\t\t键盘按键【ESC】或者【Q】- 退出程序\n"
		"\t\t键盘按键【1】- 使用椭圆(Elliptic)结构元素\n"
		"\t\t键盘按键【2】- 使用矩形(Rectangle )结构元素\n"
		"\t\t键盘按键【3】- 使用十字型(Cross-shaped)结构元素\n"
		"\t\t键盘按键【空格SPACE】- 在矩形、椭圆、十字形结构元素中循环\n"	);
}

******************放大缩小图像函数resize()*************

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

//-----------------------------------【main( )函数】--------------------------------------------
//		描述：控制台应用程序的入口函数，我们的程序从这里开始
//-----------------------------------------------------------------------------------------------
int main( )
{
	//载入原始图
	Mat srcImage = imread("1.jpg");  //工程目录下应该有一张名为1.jpg的素材图
	Mat tmpImage,dstImage1,dstImage2;//临时变量和目标图的定义
	tmpImage=srcImage;//将原始图赋给临时变量

	//显示原始图
	imshow("【原始图】", srcImage);

	//进行尺寸调整操作
	resize(tmpImage,dstImage1,Size( tmpImage.cols/2, tmpImage.rows/2 ),(0,0),(0,0),3);
	resize(tmpImage,dstImage2,Size( tmpImage.cols*2, tmpImage.rows*2 ),(0,0),(0,0),3);

	//显示效果图
	imshow("【效果图】之一", dstImage1);
	imshow("【效果图】之二", dstImage2);

	waitKey(0);
	return 0;
}

****************图像金字塔相关*******************

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;



//-----------------------------------【main( )函数】------------------------------------------
//		描述：控制台应用程序的入口函数，我们的程序从这里开始
//-----------------------------------------------------------------------------------------------
int main( )
{
	//载入原始图
	Mat srcImage = imread("1.jpg");  //工程目录下应该有一张名为1.jpg的素材图
	Mat tmpImage,dstImage;//临时变量和目标图的定义
	tmpImage=srcImage;//将原始图赋给临时变量

	//显示原始图
	imshow("【原始图】", srcImage);
	//进行向上取样操作
	pyrUp( tmpImage, dstImage, Size( tmpImage.cols*2, tmpImage.rows*2 ) );
	//显示效果图
	imshow("【效果图】", dstImage);

	waitKey(0);

	return 0;
}


#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

//-----------------------------------【main( )函数】-----------------------------------------
//		描述：控制台应用程序的入口函数，我们的程序从这里开始
//-----------------------------------------------------------------------------------------------
int main( )
{
	//载入原始图
	Mat srcImage = imread("1.jpg");  //工程目录下应该有一张名为1.jpg的素材图
	Mat tmpImage,dstImage;//临时变量和目标图的定义
	tmpImage=srcImage;//将原始图赋给临时变量

	//显示原始图
	imshow("【原始图】", srcImage);
	//进行向下取样操作
	pyrDown( tmpImage, dstImage, Size( tmpImage.cols/2, tmpImage.rows/2 ) );
	//显示效果图
	imshow("【效果图】", dstImage);

	waitKey(0);

	return 0;
}
 ***************  基本阈值操作  ******************

 #include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
using namespace cv;
using namespace std;

//-----------------------------------【宏定义部分】--------------------------------------------
//		描述：定义一些辅助宏
//------------------------------------------------------------------------------------------------
#define WINDOW_NAME "【程序窗口】"        //为窗口标题定义的宏


//-----------------------------------【全局变量声明部分】--------------------------------------
//		描述：全局变量的声明
//-----------------------------------------------------------------------------------------------
int g_nThresholdValue = 100;
int g_nThresholdType = 3;
Mat g_srcImage, g_grayImage, g_dstImage;

//-----------------------------------【全局函数声明部分】--------------------------------------
//		描述：全局函数的声明
//-----------------------------------------------------------------------------------------------
static void ShowHelpText( );//输出帮助文字
void on_Threshold( int, void* );//回调函数


//-----------------------------------【main( )函数】--------------------------------------------
//		描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-----------------------------------------------------------------------------------------------
int main( )
{
	//【0】改变console字体颜色
	system("color 1F");

	//【0】显示欢迎和帮助文字
	ShowHelpText( );

	//【1】读入源图片
	g_srcImage = imread("1.jpg");
	if(!g_srcImage.data ) { printf("读取图片错误，请确定目录下是否有imread函数指定的图片存在~！ \n"); return false; }
	imshow("原始图",g_srcImage);

	//【2】存留一份原图的灰度图
	cvtColor( g_srcImage, g_grayImage, COLOR_RGB2GRAY );

	//【3】创建窗口并显示原始图
	namedWindow( WINDOW_NAME, WINDOW_AUTOSIZE );

	//【4】创建滑动条来控制阈值
	createTrackbar( "模式",
		WINDOW_NAME, &g_nThresholdType,
		4, on_Threshold );

	createTrackbar( "参数值",
		WINDOW_NAME, &g_nThresholdValue,
		255, on_Threshold );

	//【5】初始化自定义的阈值回调函数
	on_Threshold( 0, 0 );

	// 【6】轮询等待用户按键，如果ESC键按下则退出程序
	while(1)
	{
		int key;
		key = waitKey( 20 );
		if( (char)key == 27 ){ break; }
	}

}

//-----------------------------------【on_Threshold( )函数】------------------------------------
//		描述：自定义的阈值回调函数
//-----------------------------------------------------------------------------------------------
void on_Threshold( int, void* )
{
	//调用阈值函数
	threshold(g_grayImage,g_dstImage,g_nThresholdValue,255,g_nThresholdType);

	//更新效果图
	imshow( WINDOW_NAME, g_dstImage );
}

//-----------------------------------【ShowHelpText( )函数】----------------------------------
//      描述：输出一些帮助信息
//----------------------------------------------------------------------------------------------
static void ShowHelpText()
{
	//输出欢迎信息和OpenCV版本
	printf("\n\n\t\t\t非常感谢购买《OpenCV3编程入门》一书！\n");
	printf("\n\n\t\t\t此为本书OpenCV3版的第55个配套示例程序\n");
	printf("\n\n\t\t\t   当前使用的OpenCV版本为：" CV_VERSION );
	printf("\n\n  ----------------------------------------------------------------------------\n");

	//输出一些帮助信息
	printf(	"\n\t欢迎来到【基本阈值操作】示例程序~\n\n");
	printf(	"\n\t按键操作说明: \n\n"
		"\t\t键盘按键【ESC】- 退出程序\n"
		"\t\t滚动条模式0- 二进制阈值\n"
		"\t\t滚动条模式1- 反二进制阈值\n"
		"\t\t滚动条模式2- 截断阈值\n"
		"\t\t滚动条模式3- 反阈值化为0\n"
		"\t\t滚动条模式4- 阈值化为0\n"  );
}

****************canny算子使用*************
#include <opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
using namespace cv;


//-----------------------------------【main( )函数】-------------------------------------------
//            描述：控制台应用程序的入口函数，我们的程序从这里开始
//-----------------------------------------------------------------------------------------------
int main( )
{
	//载入原始图
	Mat srcImage = imread("1.jpg");  //工程目录下应该有一张名为1.jpg的素材图
	Mat srcImage1=srcImage.clone();

	//显示原始图
	imshow("【原始图】Canny边缘检测", srcImage);

	//----------------------------------------------------------------------------------
	//	一、最简单的canny用法，拿到原图后直接用。
	//	注意：此方法在OpenCV2中可用，在OpenCV3中已失效
	//----------------------------------------------------------------------------------
 	//Canny( srcImage, srcImage, 150, 100,3 );
	//imshow("【效果图】Canny边缘检测", srcImage);


	//----------------------------------------------------------------------------------
	//	二、高阶的canny用法，转成灰度图，降噪，用canny，最后将得到的边缘作为掩码，拷贝原图到效果图上，得到彩色的边缘图
	//----------------------------------------------------------------------------------
	Mat dstImage,edge,grayImage;

	// 【1】创建与src同类型和大小的矩阵(dst)
	dstImage.create( srcImage1.size(), srcImage1.type() );

	// 【2】将原图像转换为灰度图像
	cvtColor( srcImage1, grayImage, COLOR_BGR2GRAY );

	// 【3】先用使用 3x3内核来降噪
	blur( grayImage, edge, Size(3,3) );

	// 【4】运行Canny算子
	Canny( edge, edge, 3, 9,3 );

	//【5】将g_dstImage内的所有元素设置为0
	dstImage = Scalar::all(0);

	//【6】使用Canny算子输出的边缘图g_cannyDetectedEdges作为掩码，来将原图g_srcImage拷到目标图g_dstImage中
	srcImage1.copyTo( dstImage, edge);

	//【7】显示效果图
	imshow("【效果图】Canny边缘检测2", dstImage);


	waitKey(0);

	return 0;
}
****************sobel算子使用**************
#include <opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

//-----------------------------------【命名空间声明部分】---------------------------------------
//            描述：包含程序所使用的命名空间
//-----------------------------------------------------------------------------------------------
using namespace cv;
//-----------------------------------【main( )函数】--------------------------------------------
//            描述：控制台应用程序的入口函数，我们的程序从这里开始
//-----------------------------------------------------------------------------------------------
int main( )
{
	//【0】创建 grad_x 和 grad_y 矩阵
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y,dst;

	//【1】载入原始图
	Mat src = imread("1.jpg");  //工程目录下应该有一张名为1.jpg的素材图

	//【2】显示原始图
	imshow("【原始图】sobel边缘检测", src);

	//【3】求 X方向梯度
	Sobel( src, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT );
	convertScaleAbs( grad_x, abs_grad_x );
	imshow("【效果图】 X方向Sobel", abs_grad_x);

	//【4】求Y方向梯度
	Sobel( src, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT );
	convertScaleAbs( grad_y, abs_grad_y );
	imshow("【效果图】Y方向Sobel", abs_grad_y);

	//【5】合并梯度(近似)
	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst );
	imshow("【效果图】整体方向Sobel", dst);

	waitKey(0);
	return 0;
}

*****************Laplacian函数用法*************

#include <opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
using namespace cv;


//-----------------------------------【main( )函数】--------------------------------------------
//            描述：控制台应用程序的入口函数，我们的程序从这里开始
//-----------------------------------------------------------------------------------------------
int main( )
{
	//【0】变量的定义
	Mat src,src_gray,dst, abs_dst;

	//【1】载入原始图
	src = imread("1.jpg");  //工程目录下应该有一张名为1.jpg的素材图

	//【2】显示原始图
	imshow("【原始图】图像Laplace变换", src);

	//【3】使用高斯滤波消除噪声
	GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );

	//【4】转换为灰度图
	cvtColor( src, src_gray, COLOR_RGB2GRAY );

	//【5】使用Laplace函数
	Laplacian( src_gray, dst, CV_16S, 3, 1, 0, BORDER_DEFAULT );

	//【6】计算绝对值，并将结果转换成8位
	convertScaleAbs( dst, abs_dst );

	//【7】显示效果图
	imshow( "【效果图】图像Laplace变换", abs_dst );

	waitKey(0);

	return 0;
}

***************Scharr滤波器使用**********

#include <opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
using namespace cv;



//-----------------------------------【main( )函数】--------------------------------------------
//            描述：控制台应用程序的入口函数，我们的程序从这里开始
//-----------------------------------------------------------------------------------------------
int main( )
{
	//【0】创建 grad_x 和 grad_y 矩阵
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y,dst;

	//【1】载入原始图
	Mat src = imread("1.jpg");  //工程目录下应该有一张名为1.jpg的素材图

	//【2】显示原始图
	imshow("【原始图】Scharr滤波器", src);

	//【3】求 X方向梯度
	Scharr( src, grad_x, CV_16S, 1, 0, 1, 0, BORDER_DEFAULT );
	convertScaleAbs( grad_x, abs_grad_x );
	imshow("【效果图】 X方向Scharr", abs_grad_x);

	//【4】求Y方向梯度
	Scharr( src, grad_y, CV_16S, 0, 1, 1, 0, BORDER_DEFAULT );
	convertScaleAbs( grad_y, abs_grad_y );
	imshow("【效果图】Y方向Scharr", abs_grad_y);

	//【5】合并梯度(近似)
	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst );

	//【6】显示效果图
	imshow("【效果图】合并梯度后Scharr", dst);

	waitKey(0);
	return 0;
}

***************标准霍夫变换**********HoughLines()
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
using namespace std;


//-----------------------------------【main( )函数】--------------------------------------------
//		描述：控制台应用程序的入口函数，我们的程序从这里开始
//-----------------------------------------------------------------------------------------------
int main( )
{
	//【1】载入原始图和Mat变量定义
	Mat srcImage = imread("1.jpg");  //工程目录下应该有一张名为1.jpg的素材图
	Mat midImage,dstImage;//临时变量和目标图的定义

	//【2】进行边缘检测和转化为灰度图
	Canny(srcImage, midImage, 50, 200, 3);//进行一此canny边缘检测
	cvtColor(midImage,dstImage, COLOR_GRAY2BGR);//转化边缘检测后的图为灰度图

	//【3】进行霍夫线变换
	vector<Vec2f> lines;//定义一个矢量结构lines用于存放得到的线段矢量集合
	HoughLines(midImage, lines, 1, CV_PI/180, 150, 0, 0 );

	//【4】依次在图中绘制出每条线段
	for( size_t i = 0; i < lines.size(); i++ )
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000*(-b));
		pt1.y = cvRound(y0 + 1000*(a));
		pt2.x = cvRound(x0 - 1000*(-b));
		pt2.y = cvRound(y0 - 1000*(a));
		//此句代码的OpenCV2版为:
		//line( dstImage, pt1, pt2, Scalar(55,100,195), 1, CV_AA);
		//此句代码的OpenCV3版为:
		line( dstImage, pt1, pt2, Scalar(55,100,195), 1, LINE_AA);
	}

	//【5】显示原始图
	imshow("【原始图】", srcImage);

	//【6】边缘检测后的图
	imshow("【边缘检测后的图】", midImage);

	//【7】显示效果图
	imshow("【效果图】", dstImage);

	waitKey(0);

	return 0;
}
******************累计霍夫变换************HoughLinesP()
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
using namespace std;


//-----------------------------------【main( )函数】--------------------------------------------
//		描述：控制台应用程序的入口函数，我们的程序从这里开始
//-------------------------------------------------------------------------------------------------
int main( )
{
	//【1】载入原始图和Mat变量定义
	Mat srcImage = imread("1.jpg");  //工程目录下应该有一张名为1.jpg的素材图
	Mat midImage,dstImage;//临时变量和目标图的定义

	//【2】进行边缘检测和转化为灰度图
	Canny(srcImage, midImage, 50, 200, 3);//进行一此canny边缘检测
	cvtColor(midImage,dstImage, COLOR_GRAY2BGR);//转化边缘检测后的图为灰度图

	//【3】进行霍夫线变换
	vector<Vec4i> lines;//定义一个矢量结构lines用于存放得到的线段矢量集合
	HoughLinesP(midImage, lines, 1, CV_PI/180, 80, 50, 10 );

	//【4】依次在图中绘制出每条线段
	for( size_t i = 0; i < lines.size(); i++ )
	{
		Vec4i l = lines[i];
		//此句代码的OpenCV2版为：
		//line( dstImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(186,88,255), 1, CV_AA);
		//此句代码的OpenCV3版为：
		line( dstImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(186,88,255), 1, LINE_AA);
	}

	//【5】显示原始图
	imshow("【原始图】", srcImage);

	//【6】边缘检测后的图
	imshow("【边缘检测后的图】", midImage);

	//【7】显示效果图
	imshow("【效果图】", dstImage);

	waitKey(0);

	return 0;
}



****************霍夫圆变换***********HoughCircles()
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
using namespace std;

//-----------------------------------【main( )函数】--------------------------------------------
//		描述：控制台应用程序的入口函数，我们的程序从这里开始
//-----------------------------------------------------------------------------------------------
int main( )
{
	//【1】载入原始图、Mat变量定义
	Mat srcImage = imread("1.jpg");  //工程目录下应该有一张名为1.jpg的素材图
	Mat midImage,dstImage;//临时变量和目标图的定义

	//【2】显示原始图
	imshow("【原始图】", srcImage);

	//【3】转为灰度图并进行图像平滑
	cvtColor(srcImage,midImage, COLOR_BGR2GRAY);//转化边缘检测后的图为灰度图
	GaussianBlur( midImage, midImage, Size(9, 9), 2, 2 );

	//【4】进行霍夫圆变换
	vector<Vec3f> circles;
	HoughCircles( midImage, circles, HOUGH_GRADIENT,1.5, 10, 200, 100, 0, 0 );

	//【5】依次在图中绘制出圆
	for( size_t i = 0; i < circles.size(); i++ )
	{
		//参数定义
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		//绘制圆心
		circle( srcImage, center, 3, Scalar(0,255,0), -1, 8, 0 );
		//绘制圆轮廓
		circle( srcImage, center, radius, Scalar(155,50,255), 3, 8, 0 );
	}

	//【6】显示效果图
	imshow("【效果图】", srcImage);

	waitKey(0);

	return 0;
}

*******************重映射*************
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
using namespace cv;


//-----------------------------------【main( )函数】--------------------------------------------
//          描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-----------------------------------------------------------------------------------------------
int main(  )
{
	//【0】变量定义
	Mat srcImage, dstImage;
	Mat map_x, map_y;

	//【1】载入原始图
	srcImage = imread( "1.jpg", 1 );
	if(!srcImage.data ) { printf("读取图片错误，请确定目录下是否有imread函数指定的图片存在~！ \n"); return false; }
	imshow("原始图",srcImage);

	//【2】创建和原始图一样的效果图，x重映射图，y重映射图
	dstImage.create( srcImage.size(), srcImage.type() );
	map_x.create( srcImage.size(), CV_32FC1 );
	map_y.create( srcImage.size(), CV_32FC1 );

	//【3】双层循环，遍历每一个像素点，改变map_x & map_y的值
	for( int j = 0; j < srcImage.rows;j++)
	{
		for( int i = 0; i < srcImage.cols;i++)
		{
			//改变map_x & map_y的值.
			map_x.at<float>(j,i) = static_cast<float>(i);
			map_y.at<float>(j,i) = static_cast<float>(srcImage.rows - j);
		}
	}

	//【4】进行重映射操作
	//此句代码的OpenCV2版为：
	//remap( srcImage, dstImage, map_x, map_y, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0, 0) );
	//此句代码的OpenCV3版为：
	remap( srcImage, dstImage, map_x, map_y, INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0, 0) );

	//【5】显示效果图
	imshow( "【程序窗口】", dstImage );
	waitKey();

	return 0;
}

****************仿射变换****************
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;


//-----------------------------------【宏定义部分】--------------------------------------------
//		描述：定义一些辅助宏
//------------------------------------------------------------------------------------------------
#define WINDOW_NAME1 "【原始图窗口】"					//为窗口标题定义的宏
#define WINDOW_NAME2 "【经过Warp后的图像】"        //为窗口标题定义的宏
#define WINDOW_NAME3 "【经过Warp和Rotate后的图像】"        //为窗口标题定义的宏



//-----------------------------------【全局函数声明部分】--------------------------------------
//		描述：全局函数的声明
//-----------------------------------------------------------------------------------------------
static void ShowHelpText( );


//-----------------------------------【main( )函数】--------------------------------------------
//		描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-----------------------------------------------------------------------------------------------
int main(  )
{
	//【0】改变console字体颜色
	system("color 1F");

	//【0】显示欢迎和帮助文字
	ShowHelpText( );

	//【1】参数准备
	//定义两组点，代表两个三角形
	Point2f srcTriangle[3];
	Point2f dstTriangle[3];
	//定义一些Mat变量
	Mat rotMat( 2, 3, CV_32FC1 );
	Mat warpMat( 2, 3, CV_32FC1 );
	Mat srcImage, dstImage_warp, dstImage_warp_rotate;

	//【2】加载源图像并作一些初始化
	srcImage = imread( "1.jpg", 1 );
	if(!srcImage.data ) { printf("读取图片错误，请确定目录下是否有imread函数指定的图片存在~！ \n"); return false; }
	// 设置目标图像的大小和类型与源图像一致
	dstImage_warp = Mat::zeros( srcImage.rows, srcImage.cols, srcImage.type() );

	//【3】设置源图像和目标图像上的三组点以计算仿射变换
	srcTriangle[0] = Point2f( 0,0 );
	srcTriangle[1] = Point2f( static_cast<float>(srcImage.cols - 1), 0 );
	srcTriangle[2] = Point2f( 0, static_cast<float>(srcImage.rows - 1 ));

	dstTriangle[0] = Point2f( static_cast<float>(srcImage.cols*0.0), static_cast<float>(srcImage.rows*0.33));
	dstTriangle[1] = Point2f( static_cast<float>(srcImage.cols*0.65), static_cast<float>(srcImage.rows*0.35));
	dstTriangle[2] = Point2f( static_cast<float>(srcImage.cols*0.15), static_cast<float>(srcImage.rows*0.6));

	//【4】求得仿射变换
	warpMat = getAffineTransform( srcTriangle, dstTriangle );

	//【5】对源图像应用刚刚求得的仿射变换
	warpAffine( srcImage, dstImage_warp, warpMat, dstImage_warp.size() );

	//【6】对图像进行缩放后再旋转
	// 计算绕图像中点顺时针旋转50度缩放因子为0.6的旋转矩阵
	Point center = Point( dstImage_warp.cols/2, dstImage_warp.rows/2 );
	double angle = -50.0;
	double scale = 0.6;
	// 通过上面的旋转细节信息求得旋转矩阵
	rotMat = getRotationMatrix2D( center, angle, scale );
	// 旋转已缩放后的图像
	warpAffine( dstImage_warp, dstImage_warp_rotate, rotMat, dstImage_warp.size() );


	//【7】显示结果
	imshow( WINDOW_NAME1, srcImage );
	imshow( WINDOW_NAME2, dstImage_warp );
	imshow( WINDOW_NAME3, dstImage_warp_rotate );

	// 等待用户按任意按键退出程序
	waitKey(0);

	return 0;
}


//-----------------------------------【ShowHelpText( )函数】----------------------------------
//      描述：输出一些帮助信息
//----------------------------------------------------------------------------------------------
static void ShowHelpText()
{

	//输出欢迎信息和OpenCV版本
	printf("\n\n\t\t\t非常感谢购买《OpenCV3编程入门》一书！\n");
	printf("\n\n\t\t\t此为本书OpenCV3版的第67个配套示例程序\n");
	printf("\n\n\t\t\t   当前使用的OpenCV版本为：" CV_VERSION );
	printf("\n\n  ----------------------------------------------------------------------------\n");

	//输出一些帮助信息
	printf(   "\n\n\t\t欢迎来到仿射变换综合示例程序\n\n");
	printf(  "\t\t键盘按键【ESC】- 退出程序\n"  );
}

*************查找并绘制轮廓***************
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;
using namespace std;

//-----------------------------------【main( )函数】--------------------------------------------

//		描述：控制台应用程序的入口函数，我们的程序从这里开始
//-------------------------------------------------------------------------------------------------
int main( int argc, char** argv )
{
	// 【1】载入原始图，且必须以二值图模式载入
	Mat srcImage=imread("1.jpg", 0);
	imshow("原始图",srcImage);

	//【2】初始化结果图
	Mat dstImage = Mat::zeros(srcImage.rows, srcImage.cols, CV_8UC3);

	//【3】srcImage取大于阈值119的那部分
	srcImage = srcImage > 119;
	imshow( "取阈值后的原始图", srcImage );

	//【4】定义轮廓和层次结构
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	//【5】查找轮廓
	//此句代码的OpenCV2版为：
	//findContours( srcImage, contours, hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
	//此句代码的OpenCV3版为：
	findContours( srcImage, contours, hierarchy,RETR_CCOMP, CHAIN_APPROX_SIMPLE );

	// 【6】遍历所有顶层的轮廓， 以随机颜色绘制出每个连接组件颜色
	int index = 0;
	for( ; index >= 0; index = hierarchy[index][0] )
	{
		Scalar color( rand()&255, rand()&255, rand()&255 );
		//此句代码的OpenCV2版为：
		//drawContours( dstImage, contours, index, color, CV_FILLED, 8, hierarchy );
		//此句代码的OpenCV3版为：
		drawContours( dstImage, contours, index, color, FILLED, 8, hierarchy );
	}

	//【7】显示最后的轮廓图
	imshow( "轮廓图", dstImage );

	waitKey(0);

}
*************寻找和绘制物体凸包*************

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;


//-----------------------------------【宏定义部分】--------------------------------------------
//  描述：定义一些辅助宏
//------------------------------------------------------------------------------------------------
#define WINDOW_NAME1 "【原始图窗口】"					//为窗口标题定义的宏
#define WINDOW_NAME2 "【效果图窗口】"					//为窗口标题定义的宏



//-----------------------------------【全局变量声明部分】--------------------------------------
//  描述：全局变量的声明
//-----------------------------------------------------------------------------------------------
Mat g_srcImage; Mat g_grayImage;
int g_nThresh = 50;
int g_maxThresh = 255;
RNG g_rng(12345);
Mat srcImage_copy = g_srcImage.clone();
Mat g_thresholdImage_output;
vector<vector<Point> > g_vContours;
vector<Vec4i> g_vHierarchy;


//-----------------------------------【全局函数声明部分】--------------------------------------
//   描述：全局函数的声明
//-----------------------------------------------------------------------------------------------
static void ShowHelpText( );
void on_ThreshChange(int, void* );
void ShowHelpText();

//-----------------------------------【main( )函数】------------------------------------------
//   描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-----------------------------------------------------------------------------------------------
int main(  )
{
	system("color 3F");
	ShowHelpText();

	// 加载源图像
	g_srcImage = imread( "1.jpg", 1 );

	// 将原图转换成灰度图并进行模糊降
	cvtColor( g_srcImage, g_grayImage, COLOR_BGR2GRAY );
	blur( g_grayImage, g_grayImage, Size(3,3) );

	// 创建原图窗口并显示
	namedWindow( WINDOW_NAME1, WINDOW_AUTOSIZE );
	imshow( WINDOW_NAME1, g_srcImage );

	//创建滚动条
	createTrackbar( " 阈值:", WINDOW_NAME1, &g_nThresh, g_maxThresh, on_ThreshChange );
	on_ThreshChange( 0, 0 );//调用一次进行初始化

	waitKey(0);
	return(0);
}

//-----------------------------------【thresh_callback( )函数】----------------------------------
//      描述：回调函数
//----------------------------------------------------------------------------------------------
void on_ThreshChange(int, void* )
{
	// 对图像进行二值化，控制阈值
	threshold( g_grayImage, g_thresholdImage_output, g_nThresh, 255, THRESH_BINARY );

	// 寻找轮廓
	findContours( g_thresholdImage_output, g_vContours, g_vHierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );

	// 遍历每个轮廓，寻找其凸包
	vector<vector<Point> >hull( g_vContours.size() );
	for( unsigned int i = 0; i < g_vContours.size(); i++ )
	{
		convexHull( Mat(g_vContours[i]), hull[i], false );
	}

	// 绘出轮廓及其凸包
	Mat drawing = Mat::zeros( g_thresholdImage_output.size(), CV_8UC3 );
	for(unsigned  int i = 0; i< g_vContours.size(); i++ )
	{
		Scalar color = Scalar( g_rng.uniform(0, 255), g_rng.uniform(0,255), g_rng.uniform(0,255) );
		drawContours( drawing, g_vContours, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
		drawContours( drawing, hull, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
	}

	// 显示效果图
	imshow( WINDOW_NAME2, drawing );
}


//-----------------------------------【ShowHelpText( )函数】-----------------------------
//		 描述：输出一些帮助信息
//----------------------------------------------------------------------------------------------
void ShowHelpText()
{
	//输出欢迎信息和OpenCV版本
	printf("\n\n\t\t\t非常感谢购买《OpenCV3编程入门》一书！\n");
	printf("\n\n\t\t\t此为本书OpenCV3版的第72个配套示例程序\n");
	printf("\n\n\t\t\t   当前使用的OpenCV版本为：" CV_VERSION );
	printf("\n\n  ----------------------------------------------------------------------------\n");
}

*************外部矩形边界boundingRect()************
最小包围矩形minAreaRect()
最小包围圆形minEnclosingCircle()
用椭圆拟合二维点集fitEllipse()
逼近多边形曲线approxPolyDP()


******创建包围轮廓的矩形边界  先创建一些随机点再画包围的轮廓
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;
using namespace std;



//-----------------------------------【ShowHelpText( )函数】-----------------------------
//          描述：输出一些帮助信息
//----------------------------------------------------------------------------------------------
static void ShowHelpText()
{

	//输出欢迎信息和OpenCV版本
	printf("\n\n\t\t\t非常感谢购买《OpenCV3编程入门》一书！\n");
	printf("\n\n\t\t\t此为本书OpenCV3版的第73个配套示例程序\n");
	printf("\n\n\t\t\t   当前使用的OpenCV版本为：" CV_VERSION );
	printf("\n\n  ----------------------------------------------------------------------------\n");

	//输出一些帮助信息
	printf("\n\n\n\t\t\t欢迎来到【矩形包围示例】示例程序~\n\n");
	printf("\n\n\t按键操作说明: \n\n"
		"\t\t键盘按键【ESC】、【Q】、【q】- 退出程序\n\n"
		"\t\t键盘按键任意键 - 重新生成随机点，并寻找最小面积的包围矩形\n" );
}

int main(  )
{
	//改变console字体颜色
	system("color 1F");

	//显示帮助文字
	ShowHelpText();

	//初始化变量和随机值
	Mat image(600, 600, CV_8UC3);
	RNG& rng = theRNG();

	//循环，按下ESC,Q,q键程序退出，否则有键按下便一直更新
	while(1)
	{
		//参数初始化
		int count = rng.uniform(3, 103);//随机生成点的数量
		vector<Point> points;//点值

		//随机生成点坐标
		for(int  i = 0; i < count; i++ )
		{

			Point point;
			point.x = rng.uniform(image.cols/4, image.cols*3/4);
			point.y = rng.uniform(image.rows/4, image.rows*3/4);

			points.push_back(point);
		}

		//对给定的 2D 点集，寻找最小面积的包围矩形
		RotatedRect box = minAreaRect(Mat(points));
		Point2f vertex[4];
		box.points(vertex);

		//绘制出随机颜色的点
		image = Scalar::all(0);
		for( int i = 0; i < count; i++ )
			circle( image, points[i], 3, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), FILLED, LINE_AA );


		//绘制出最小面积的包围矩形
		for( int i = 0; i < 4; i++ )
			line(image, vertex[i], vertex[(i+1)%4], Scalar(100, 200, 211), 2, LINE_AA);

		//显示窗口
		imshow( "矩形包围示例", image );

		//按下ESC,Q,或者q，程序退出
		char key = (char)waitKey();
		if( key == 27 || key == 'q' || key == 'Q' ) // 'ESC'
			break;
	}

	return 0;
}


**********创建包围轮廓的圆形边界    先随机生成随机点，再画包围轮廓的圆形边界
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;
using namespace std;

//-----------------------------------【ShowHelpText( )函数】----------------------------------
//          描述：输出一些帮助信息
//----------------------------------------------------------------------------------------------
static void ShowHelpText()
{

	//输出欢迎信息和OpenCV版本
	printf("\n\n\t\t\t非常感谢购买《OpenCV3编程入门》一书！\n");
	printf("\n\n\t\t\t此为本书OpenCV3版的第13个配套示例程序\n");
	printf("\n\n\t\t\t   当前使用的OpenCV版本为：" CV_VERSION );
	printf("\n\n  ----------------------------------------------------------------------------\n");

	//输出一些帮助信息
	printf("\n\n\t\t\t欢迎来到【寻找最小面积的包围圆】示例程序~\n");
	printf("\n\n\t按键操作说明: \n\n"
		"\t\t键盘按键【ESC】、【Q】、【q】- 退出程序\n\n"
		"\t\t键盘按键任意键 - 重新生成随机点，并寻找最小面积的包围圆\n" );
}

int main(  )
{
	//改变console字体颜色
	system("color 1F");

	//显示帮助文字
	ShowHelpText();

	//初始化变量和随机值
	Mat image(600, 600, CV_8UC3);
	RNG& rng = theRNG();

	//循环，按下ESC,Q,q键程序退出，否则有键按下便一直更新
	while(1)
	{
		//参数初始化
		int count = rng.uniform(3, 103);//随机生成点的数量
		vector<Point> points;//点值

		//随机生成点坐标
		for(int  i = 0; i < count; i++ )
		{

			Point point;
			point.x = rng.uniform(image.cols/4, image.cols*3/4);
			point.y = rng.uniform(image.rows/4, image.rows*3/4);

			points.push_back(point);
		}

		//对给定的 2D 点集，寻找最小面积的包围圆
		Point2f center;
		float radius = 0;
		minEnclosingCircle(Mat(points), center, radius);

		//绘制出随机颜色的点
		image = Scalar::all(0);
		for( int i = 0; i < count; i++ )
			circle( image, points[i], 3, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), FILLED, LINE_AA );


		//绘制出最小面积的包围圆
		circle(image, center, cvRound(radius), Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), 2, LINE_AA);

		//显示窗口
		imshow( "圆形包围示例", image );

		//按下ESC,Q,或者q，程序退出
		char key = (char)waitKey();
		if( key == 27 || key == 'q' || key == 'Q' ) // 'ESC'
			break;
	}

	return 0;
}

***************矩的计算 moments()函数**************
计算轮廓面积 contourArea()
计算轮廓长度 arcLength()

*************查找和绘制图像轮廓矩************

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;


//-----------------------------------【宏定义部分】--------------------------------------------
//		描述：定义一些辅助宏
//------------------------------------------------------------------------------------------------
#define WINDOW_NAME1 "【原始图】"					//为窗口标题定义的宏
#define WINDOW_NAME2 "【图像轮廓】"        //为窗口标题定义的宏


//-----------------------------------【全局变量声明部分】--------------------------------------
//		描述：全局变量的声明
//-----------------------------------------------------------------------------------------------
Mat g_srcImage; Mat g_grayImage;
int g_nThresh = 100;
int g_nMaxThresh = 255;
RNG g_rng(12345);
Mat g_cannyMat_output;
vector<vector<Point> > g_vContours;
vector<Vec4i> g_vHierarchy;

//-----------------------------------【全局变量声明部分】--------------------------------------
//		描述：全局变量的声明
//-----------------------------------------------------------------------------------------------
void on_ThreshChange(int, void* );
static void ShowHelpText( );

//-----------------------------------【main( )函数】--------------------------------------------
//		描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-----------------------------------------------------------------------------------------------
int main( int argc, char** argv )
{
	//【0】改变console字体颜色
	system("color 9F");

	ShowHelpText();
	// 读入原图像, 返回3通道图像数据
	g_srcImage = imread( "1.jpg", 1 );

	// 把原图像转化成灰度图像并进行平滑
	cvtColor( g_srcImage, g_grayImage, COLOR_BGR2GRAY );
	blur( g_grayImage, g_grayImage, Size(3,3) );

	// 创建新窗口
	namedWindow( WINDOW_NAME1, WINDOW_AUTOSIZE );
	imshow( WINDOW_NAME1, g_srcImage );

	//创建滚动条并进行初始化
	createTrackbar( " 阈值", WINDOW_NAME1, &g_nThresh, g_nMaxThresh, on_ThreshChange );
	on_ThreshChange( 0, 0 );

	waitKey(0);
	return(0);
}

//-----------------------------------【on_ThreshChange( )函数】-------------------------------
//		描述：回调函数
//-----------------------------------------------------------------------------------------------
void on_ThreshChange(int, void* )
{
	// 使用Canndy检测边缘
	Canny( g_grayImage, g_cannyMat_output, g_nThresh, g_nThresh*2, 3 );

	// 找到轮廓
	findContours( g_cannyMat_output, g_vContours, g_vHierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );

	// 计算矩
	vector<Moments> mu(g_vContours.size() );
	for(unsigned int i = 0; i < g_vContours.size(); i++ )
	{ mu[i] = moments( g_vContours[i], false ); }

	//  计算中心矩
	vector<Point2f> mc( g_vContours.size() );
	for( unsigned int i = 0; i < g_vContours.size(); i++ )
	{ mc[i] = Point2f( static_cast<float>(mu[i].m10/mu[i].m00), static_cast<float>(mu[i].m01/mu[i].m00 )); }

	// 绘制轮廓
	Mat drawing = Mat::zeros( g_cannyMat_output.size(), CV_8UC3 );
	for( unsigned int i = 0; i< g_vContours.size(); i++ )
	{
		Scalar color = Scalar( g_rng.uniform(0, 255), g_rng.uniform(0,255), g_rng.uniform(0,255) );//随机生成颜色值
		drawContours( drawing, g_vContours, i, color, 2, 8, g_vHierarchy, 0, Point() );//绘制外层和内层轮廓
		circle( drawing, mc[i], 4, color, -1, 8, 0 );;//绘制圆
	}

	// 显示到窗口中
	namedWindow( WINDOW_NAME2, WINDOW_AUTOSIZE );
	imshow( WINDOW_NAME2, drawing );

	// 通过m00计算轮廓面积并且和OpenCV函数比较
	printf("\t 输出内容: 面积和轮廓长度\n");
	for(unsigned  int i = 0; i< g_vContours.size(); i++ )
	{
		printf(" >通过m00计算出轮廓[%d]的面积: (M_00) = %.2f \n OpenCV函数计算出的面积=%.2f , 长度: %.2f \n\n", i, mu[i].m00, contourArea(g_vContours[i]), arcLength( g_vContours[i], true ) );
		Scalar color = Scalar( g_rng.uniform(0, 255), g_rng.uniform(0,255), g_rng.uniform(0,255) );
		drawContours( drawing, g_vContours, i, color, 2, 8, g_vHierarchy, 0, Point() );
		circle( drawing, mc[i], 4, color, -1, 8, 0 );
	}
}


//-----------------------------------【ShowHelpText( )函数】-----------------------------
//		 描述：输出一些帮助信息
//----------------------------------------------------------------------------------------------
void ShowHelpText()
{
	//输出欢迎信息和OpenCV版本
	printf("\n\n\t\t\t非常感谢购买《OpenCV3编程入门》一书！\n");
	printf("\n\n\t\t\t此为本书OpenCV3版的第76个配套示例程序\n");
	printf("\n\n\t\t\t   当前使用的OpenCV版本为：" CV_VERSION );
	printf("\n\n  ----------------------------------------------------------------------------\n");
}

***********分水岭算法watershed()*************
***********图像修补inpaint()***************
计算直方图    calcHist()
找寻最值      minMaxLoc()
对比直方图    compareHist()
计算反向投影  calcBackProject()
通道复制      mixChannels()

实现模板匹配  matchTemplate()



************角点检测**********
实现Harris角点检测   CornerHarris()
Shi-Tomasi角点检测  goodFeaturesToTrack()
寻找亚像素角点      cornerSubPix()

*************特征检测与匹配*************

SURF    SurfFeatureDetector  SurfDescriporExtractor 三个类
绘制关键点 drawKeypoints()
绘制匹配点 drawMatches()


***********ORB特征提取**************
ORB  OrbFeatureDetector OrbDesciptorExtractor 三个类
