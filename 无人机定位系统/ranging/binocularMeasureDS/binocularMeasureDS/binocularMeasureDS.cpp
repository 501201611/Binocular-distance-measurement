#include "CameraDS.h"
#include "opencv2/opencv.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

using namespace cv;
using namespace std;

Size	imageSize;
int		imageWidth = 640;
int		imageHeight = 480;
bool	data_Loaded;			// 是否成功载入定标参数
Mat		Mat_Remap_X_L;			// 左视图畸变校正像素坐标映射矩阵 X
Mat		Mat_Remap_Y_L;			// 左视图畸变校正像素坐标映射矩阵 Y
Mat		Mat_Remap_X_R;			// 右视图畸变校正像素坐标映射矩阵 X
Mat		Mat_Remap_Y_R;			// 右视图畸变校正像素坐标映射矩阵 Y
Mat		Mat_Mask_Roi;			// 左视图校正后的有效区域
Rect	Roi_L;					// 左视图校正后的有效区域矩形
Rect	Roi_R;					// 右视图校正后的有效区域矩形
Mat		pointClouds;			// 点云
int		numberOfDisparies;		// 视差变化范围
cv::StereoBM	BM;				// 立体匹配 BM 方法
cv::StereoSGBM	SGBM;			// 立体匹配 SGBM 方法
Mat		disp, disp8u, disparityImage;

Mat cameraMatrixL = (Mat_<double>(3, 3));
Mat distCoeffL = (Mat_<double>(5, 1));
Mat cameraMatrixR = (Mat_<double>(3, 3));
Mat distCoeffR = (Mat_<double>(5, 1));
Mat R = (Mat_<double>(3, 3));	// R 旋转矩阵 
Mat T = (Mat_<double>(3, 1));	// T 平移矢量
Mat E, F;						// E 本征矩阵 F 基础矩阵
Mat Mat_Q;						// Q 矩阵
Mat Rl, Rr, Pl, Pr;				// 校正旋转矩阵 Rl Rr，投影矩阵 Pl Pr

// 读入摄像头定标参数并立体校正
int loadAndCalData();

// 更新 BM 参数
void updatebm();

// 更新 sgBM 参数
void updatesgbm();

// BM 法立体匹配
int  bmMatch(cv::Mat& frameLeft, cv::Mat& frameRight, cv::Mat& disparity, cv::Mat& imageLeft, cv::Mat& imageRight);

// SGBM 法立体匹配
int  sgbmMatch(cv::Mat& frameLeft, cv::Mat& frameRight, cv::Mat& disparity, cv::Mat& imageLeft, cv::Mat& imageRight);

// 获取视差图
int getDisparityImage(cv::Mat& disparity, cv::Mat& disparityImage, bool isColor);

// 获取点云
int getPointClouds(cv::Mat& disparity, cv::Mat& pointClouds);

bool wait = false;
void onMouse(int event, int x, int y,int flags, void *)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		if (pointClouds.empty())
		{
			return;
		}

		float  distance = - pointClouds.at<cv::Point3f>(y,x).z;
		if ( distance >=0 )
		{
			cout << "The distance at ( " << x << "," << y << " ) is " << distance << " cm " << endl;
		}
	}
	if (event == EVENT_RBUTTONDOWN)
	{
		wait = !wait;
	}
}


int main(int argc, char** argv)
{
	if(CCameraDS::CameraCount() < 2)
		return 0;
	
	// 创建2个摄像头对象
    CCameraDS camera1;
    CCameraDS camera2;

    //打开两个摄像头
    if(!camera1.OpenCamera(1, true, 640,480) || (!camera2.OpenCamera(2, true, 640,480)))
    {
        fprintf(stderr, "Can not open camera.");
        return -1;
    }

    namedWindow("left", 1);
    namedWindow("right", 1);
    namedWindow("disparitycolor", 1);
	setMouseCallback("disparitycolor",onMouse,0);
    loadAndCalData();
    cout << data_Loaded << endl;

	Mat frame1,frame2;//左右视图

    while (true)
    {
		//获取一帧
		frame1 = Mat(camera1.QueryFrame());
		frame2 = Mat(camera2.QueryFrame());
		
        if (frame1.empty())	break;
        if (frame2.empty())	break;
        
		Mat imageLeft, imageRight;//双目匹配后的左右视图
		//立体匹配计算计算视差
		//updatebm();
		//bmMatch(frame1, frame2, disp, imageLeft, imageRight);
		updatesgbm();
		sgbmMatch(frame1, frame2, disp, imageLeft, imageRight);
        getDisparityImage(disp, disparityImage, false);
		getPointClouds(disp, pointClouds);
        imshow("disparitycolor", disparityImage);//显示视差图
		imshow("left", imageLeft);//显示双目匹配后的左右视图
		imshow("right", imageRight);
		if (wait)
		{
			waitKey(0);
		}
		else
		{
			if(27 == waitKey(1)) break;
		}
    }
    return 0;
}


int loadAndCalData()
{
	// 读入摄像头定标参数
	FileStorage fs("calibParms.xml", FileStorage::READ);
	cout << fs.isOpened() << endl;

	if (!fs.isOpened())
	{
		data_Loaded = false;
		return 0;
	}

	cv::FileNodeIterator it = fs["imageSize"].begin();
	it >> imageSize.width >> imageSize.height;
	fs["cameraMatrixL"] >> cameraMatrixL;
	fs["cameraDistcoeffL"] >> distCoeffL;
	fs["cameraMatrixR"] >> cameraMatrixR;
	fs["cameraDistcoeffR"] >> distCoeffR;
	fs["R"] >> R;
	fs["T"] >> T;

	fs.release();
	data_Loaded = true;

	stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Mat_Q,
		CALIB_ZERO_DISPARITY,-1,imageSize,&Roi_L,&Roi_R);

	initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pl, imageSize, CV_16SC2, Mat_Remap_X_L, Mat_Remap_Y_L);
	initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_16SC2, Mat_Remap_X_R, Mat_Remap_Y_R);

	Mat_Q.at<double>(3, 2) = -Mat_Q.at<double>(3, 2);

	Mat_Mask_Roi = cv::Mat::zeros(imageHeight, imageWidth, CV_8UC1);
	cv::rectangle(Mat_Mask_Roi, Roi_L, cv::Scalar(255), -1);

	return 1;
}

void updatebm()
{
	BM.state->roi1 = Roi_L;
	BM.state->roi2 = Roi_R;
	BM.state->minDisparity = 0;
	BM.state->numberOfDisparities = 64;//64
	BM.state->SADWindowSize = 19;//19
	BM.state->textureThreshold = 10;
	BM.state->disp12MaxDiff = -1;
	BM.state->preFilterCap = 31;
	BM.state->uniquenessRatio = 25;//25
	BM.state->speckleRange = 32;
	BM.state->speckleWindowSize = 100;
}

void updatesgbm()
{
	SGBM.preFilterCap = 63;
	SGBM.SADWindowSize = 7;                              //7SAD窗口大小【1,11】
	SGBM.P1 = 8 * 3 * SGBM.SADWindowSize * SGBM.SADWindowSize ;
	SGBM.P2 = 32 * 3 * SGBM.SADWindowSize * SGBM.SADWindowSize ;
	SGBM.minDisparity = 0;//最小视差，默认为0
	SGBM.numberOfDisparities = 64;                       //64视差窗口，最大视差与最小视差之差
	SGBM.uniquenessRatio = 25;                           //25视差唯一性百分比
	SGBM.speckleWindowSize = 100;//检查视差联通区域变化度的窗口大小
	SGBM.speckleRange =16;//视差变化阈值
	SGBM.fullDP = false;
	SGBM.disp12MaxDiff = -1;
}

int  bmMatch(cv::Mat& frameLeft, cv::Mat& frameRight, cv::Mat& disparity, cv::Mat& imageLeft, cv::Mat& imageRight)
{
	// 输入检查
	if (frameLeft.empty() || frameRight.empty())
	{
		disparity = cv::Scalar(0);
		return 0;
	}
	if (imageWidth == 0 || imageHeight == 0)
	{
		return 0;
	}

	// 转换为灰度图
	cv::Mat img1proc, img2proc;
	cvtColor(frameLeft, img1proc, CV_BGR2GRAY);
	cvtColor(frameRight, img2proc, CV_BGR2GRAY);

	// 校正图像，使左右视图行对齐    
	cv::Mat img1remap, img2remap;

	if (data_Loaded)
	{
		// 对用于视差计算的画面进行校正
		remap(img1proc, img1remap, Mat_Remap_X_L, Mat_Remap_Y_L, cv::INTER_LINEAR);     
		remap(img2proc, img2remap, Mat_Remap_X_R, Mat_Remap_Y_R, cv::INTER_LINEAR);
	}
	else
	{
		img1remap = img1proc;
		img2remap = img2proc;
	}

	// 对左右视图的左边进行边界延拓，以获取与原始视图相同大小的有效视差区域
	cv::Mat img1border, img2border;
	if (numberOfDisparies != BM.state->numberOfDisparities)
		numberOfDisparies = BM.state->numberOfDisparities;
	copyMakeBorder(img1remap, img1border, 0, 0, BM.state->numberOfDisparities, 0, IPL_BORDER_REPLICATE);
	copyMakeBorder(img2remap, img2border, 0, 0, BM.state->numberOfDisparities, 0, IPL_BORDER_REPLICATE);

	// 计算视差
	cv::Mat dispBorder;
	BM(img1border, img2border, dispBorder);

	// 截取与原始画面对应的视差区域（舍去加宽的部分）
	cv::Mat disp;
	disp = dispBorder.colRange(BM.state->numberOfDisparities, img1border.cols);
	disp.copyTo(disparity, Mat_Mask_Roi);

	// 输出处理后的图像
	if (data_Loaded)
	{
		remap(frameLeft, imageLeft, Mat_Remap_X_L, Mat_Remap_Y_L, cv::INTER_LINEAR);
		rectangle(imageLeft, Roi_L, CV_RGB(0, 0, 255), 3);
	}
	else
		frameLeft.copyTo(imageLeft);

	if (data_Loaded)
		remap(frameRight, imageRight, Mat_Remap_X_R, Mat_Remap_Y_R, cv::INTER_LINEAR);
	else
		frameRight.copyTo(imageRight);
	rectangle(imageRight, Roi_R, CV_RGB(0, 0, 255), 3);

	return 1;
}


int sgbmMatch(cv::Mat& frameLeft, cv::Mat& frameRight, cv::Mat& disparity, cv::Mat& imageLeft, cv::Mat& imageRight)
{
	// 输入检查
	if (frameLeft.empty() || frameRight.empty())
	{
		disparity = cv::Scalar(0);
		return 0;
	}
	if (imageWidth == 0 || imageHeight == 0)
	{
		return 0;
	}

	// 复制图像
	cv::Mat img1proc, img2proc;
	frameLeft.copyTo(img1proc);
	frameRight.copyTo(img2proc);

	// 校正图像，使左右视图行对齐	
	cv::Mat img1remap, img2remap;
	if (data_Loaded)
	{
		// 对用于视差计算的画面进行校正
		remap(img1proc, img1remap, Mat_Remap_X_L, Mat_Remap_Y_L, cv::INTER_LINEAR);		
		remap(img2proc, img2remap, Mat_Remap_X_R, Mat_Remap_Y_R, cv::INTER_LINEAR);
	} 
	else
	{
		img1remap = img1proc;
		img2remap = img2proc;
	}

	// 对左右视图的左边进行边界延拓，以获取与原始视图相同大小的有效视差区域
	cv::Mat img1border, img2border;
	if (numberOfDisparies != SGBM.numberOfDisparities)
		numberOfDisparies = SGBM.numberOfDisparities;
	copyMakeBorder(img1remap, img1border, 0, 0, SGBM.numberOfDisparities, 0, IPL_BORDER_REPLICATE);
	copyMakeBorder(img2remap, img2border, 0, 0, SGBM.numberOfDisparities, 0, IPL_BORDER_REPLICATE);

	// 计算视差
	cv::Mat dispBorder;
	SGBM(img1border, img2border, dispBorder);

	// 截取与原始画面对应的视差区域（舍去加宽的部分）
	cv::Mat disp;
	disp = dispBorder.colRange(SGBM.numberOfDisparities, img1border.cols);	
	disp.copyTo(disparity, Mat_Mask_Roi);

	// 输出处理后的图像
	imageLeft = img1remap.clone();
	imageRight = img2remap.clone();
	rectangle(imageLeft, Roi_L, CV_RGB(0,255,0), 3);
	rectangle(imageRight, Roi_R, CV_RGB(0,255,0), 3);

	return 1;
}


int getDisparityImage(cv::Mat& disparity, cv::Mat& disparityImage, bool isColor)
{
	// 将原始视差数据的位深转换为 8 位
	cv::Mat disp8u;
	if (disparity.depth() != CV_8U)
	{
		if (disparity.depth() == CV_8S)
		{
			disparity.convertTo(disp8u, CV_8U);
		}
		else
		{
			disparity.convertTo(disp8u, CV_8U, 255 / (numberOfDisparies*16.));
		}
	}
	else
	{
		disp8u = disparity;
	}

	// 转换为灰度图像
	if (isColor)
	{
		if (disparityImage.empty() || disparityImage.type() != CV_8UC3 || disparityImage.size() != disparity.size())
		{
			disparityImage = cv::Mat::zeros(disparity.rows, disparity.cols, CV_8UC3);
		}

		for (int y = 0; y<disparity.rows; y++)
		{
			for (int x = 0; x<disparity.cols; x++)
			{
				uchar val = disp8u.at<uchar>(y, x);
				uchar r, g, b;

				if (val == 0)
					r = g = b = 0;
				else
				{
					r = 255 - val;
					g = val < 128 ? val * 2 : (uchar)((255 - val) * 2);
					b = val;
				}

				disparityImage.at<cv::Vec3b>(y, x) = cv::Vec3b(r, g, b);
			}
		}
	}
	else
	{
		disp8u.copyTo(disparityImage);
	}

	return 1;
}

//计算生成三维点云
int getPointClouds(cv::Mat& disparity, cv::Mat& pointClouds)
{
	if (disparity.empty())
	{
		return 0;
	}

	
	cv::reprojectImageTo3D(disparity, pointClouds, Mat_Q, true);
	pointClouds *= 1.6;

	return 1;
}
