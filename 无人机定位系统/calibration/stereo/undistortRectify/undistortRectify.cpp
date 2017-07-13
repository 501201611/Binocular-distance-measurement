// 立体校正，把右相机图像变换到左相机平面，输出校正后的图片

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char const *argv[])
{
	char* leftimg_directory = "..//..//imgs//leftImgs//";		// 左摄像头图像路径
	char* rightimg_directory = "..//..//imgs//rightImgs//";		// 右摄像头图像路径
	char* leftimg_filename = "left";							// 左摄像头图像名
	char* rightimg_filename = "right";							// 右摄像头图像名
	char* extension = "jpg";									// 图像格式
	char* calib_file = "..//..//parms//calibParms.xml";			// 立体校正参数文件
	char* rectifiedImg_directory = "..//..//result//rectifiedImg//";// 图像校正结果路径
	char* leftout_directory = "..//..//result//left//";			// 左摄像头图像校正结果路径
	char* rightout_directory = "..//..//result//right//";		// 右摄像头图像校正结果路径
	char* rectifiedImg_filename = "rectifiedImg";				// 校正结果图像文件名
	char* leftout_filename = "leftOut";							// 左摄像头图像校正结果文件名
	char* rightout_filename = "rightOut";						// 左摄像头图像校正结果文件名
	int num_imgs = 17;
	Size imgSize = Size(640,480);

	Mat R1, R2, P1, P2, Q;
	Mat cameraMatrixL, cameraDistcoeffL, cameraMatrixR, cameraDistcoeffR, R, T;

	cv::FileStorage fs(calib_file, cv::FileStorage::READ);
	if (!fs.isOpened())
	{
		printf("打开参数文件失败!\n");
		return -1;
	}
	fs["cameraMatrixL"] >> cameraMatrixL;
	fs["cameraDistcoeffL"] >> cameraDistcoeffL;
	fs["cameraMatrixR"] >> cameraMatrixR;
	fs["cameraDistcoeffR"] >> cameraDistcoeffR;
	fs["R"] >> R;
	fs["T"] >> T;

	fs["R1"] >> R1;//输出第一个相机的3x3矫正变换(旋转矩阵)
	fs["R2"] >> R2;
	fs["P1"] >> P1;//在第一台相机的新的坐标系统(矫正过的)输出 3x4 的投影矩阵
	fs["P2"] >> P2;
	fs["Q"] >> Q;  //深度视差映射矩阵

	cout << "参数：" << endl << endl;
	cout << "cameraMatrixL :" << endl << cameraMatrixL << endl << endl;
	cout << "cameraDistcoeffL :" << endl << cameraDistcoeffL << endl << endl;
	cout << "cameraMatrixR :" << endl << cameraMatrixR << endl << endl;
	cout << "cameraDistcoeffR :" << endl << cameraDistcoeffR << endl << endl;
	cout << "R :" << endl << R << endl << endl;
	cout << "T :" << endl << T << endl << endl;

	cv::Mat lmapx, lmapy, rmapx, rmapy;
	cv::initUndistortRectifyMap(cameraMatrixL, cameraDistcoeffL, R1, P1, imgSize, CV_32F, lmapx, lmapy);
	cv::initUndistortRectifyMap(cameraMatrixR, cameraDistcoeffR, R2, P2, imgSize, CV_32F, rmapx, rmapy);

	cv::Rect validROIL,validROIR;
	stereoRectify(cameraMatrixL, cameraDistcoeffL, cameraMatrixR, cameraDistcoeffR, imgSize, R, T,
		R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, imgSize, &validROIL, &validROIR);

	cout << "按任意键继续..." << endl;

	for (int k = 1; k <= num_imgs; k++)
	{
		char leftimg_file[100];
		char rightimg_file[100];
		sprintf(leftimg_file, "%s%s%d.%s", leftimg_directory, leftimg_filename, k, extension);
		sprintf(rightimg_file, "%s%s%d.%s", rightimg_directory, rightimg_filename, k, extension);
		Mat img1 = imread(leftimg_file, CV_LOAD_IMAGE_COLOR);
		Mat img2 = imread(rightimg_file, CV_LOAD_IMAGE_COLOR);
		if (img1.empty() || img2.empty())
		{
			printf("读入图像错误！\n");
			return -1;
		}

		cv::Mat imgU1, imgU2;

		cv::remap(img1, imgU1, lmapx, lmapy, cv::INTER_LINEAR);
		cv::remap(img2, imgU2, rmapx, rmapy, cv::INTER_LINEAR);

		char leftout_file[100];
		sprintf(leftout_file, "%s%s%d.%s", leftout_directory, leftout_filename, k, extension);
		imwrite(leftout_file, imgU1);

		char rightout_file[100];
		sprintf(rightout_file, "%s%s%d.%s", rightout_directory, rightout_filename, k, extension);
		imwrite(rightout_file, imgU2);

		cout << k << ". rectified!" << endl;


		// 显示部分

		Mat rectifiedImg;
		double sf = 1.0;
		int w = cvRound(imgSize.width * sf), h = cvRound(imgSize.height * sf);
		rectifiedImg.create(h, w * 2, CV_8UC3);

		// 画上左右图像
		Mat rectifiedImgPartL = rectifiedImg(Rect(w * 0, 0, w, h)); 
		Mat rectifiedImgPartR = rectifiedImg(Rect(w, 0, w, h));
		resize(imgU1, rectifiedImgPartL, rectifiedImgPartL.size(), 0, 0, INTER_AREA);
		resize(imgU2, rectifiedImgPartR, rectifiedImgPartR.size(), 0, 0, INTER_LINEAR);
		Rect vroiL(cvRound(validROIL.x * sf), cvRound(validROIL.y * sf), 
			cvRound(validROIL.width * sf), cvRound(validROIL.height * sf)); // 缩放有效区域
		Rect vroiR(cvRound(validROIR.x * sf), cvRound(validROIR.y * sf), 
			cvRound(validROIR.width * sf), cvRound(validROIR.height * sf));
		rectangle(rectifiedImgPartL, vroiL, Scalar(0, 0, 255), 3, 8);		// 在有效区域画上矩形  
		rectangle(rectifiedImgPartR, vroiR, Scalar(0, 0, 255), 3, 8); 

		// 画上对应的线条
		for (int i = 0; i < rectifiedImg.rows; i += 16)
			line(rectifiedImg, Point(0, i), Point(rectifiedImg.cols, i), Scalar(0, 255, 0), 1, 8);
		imshow("rectified", rectifiedImg);

		// 保存图像
		char rectifiedImg_file[100];
		sprintf(rectifiedImg_file, "%s%s%d.%s", rectifiedImg_directory, rectifiedImg_filename, k, extension);
		imwrite(rectifiedImg_file, rectifiedImg);

		waitKey(0);
	}
	
	return 0;
}