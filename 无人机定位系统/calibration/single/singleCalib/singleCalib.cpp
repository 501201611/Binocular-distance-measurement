// 对单个摄像头进行标定，输出参数文件

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>

using namespace std;
using namespace cv;

int setup_calibration(	vector<vector<Point3f>> &object_points, vector<vector<Point2f>> &image_points, int board_width, int board_height, int num_imgs, float square_size, char* imgs_directory, char* imgs_filename, char* extension);

int main(int argc, char const **argv)
{
	Size imgSize = Size(640,480);							
	int board_width = 9, board_height = 6;					
	int num_imgs = 17;										
	float square_size = 21;									

	char* imgs_directory = "..//..//imgs//leftImgs//";		// 左摄像头路径
	char* imgs_filename = "left";							// 左摄像头图片名
	char* extension = "jpg";								// 左摄像头图片格式
	char* out_file = "..//..//parms//leftParms.xml";		// 左摄像头参数路径及文件名

	//char* imgs_directory = "..//..//imgs//rightImgs//";		// 右摄像头路径
	//char* imgs_filename = "right";							// 右摄像头图片名
	//char* extension = "jpg";								// 右摄像头图片格式
	//char* out_file = "..//..//parms//rightParms.xml";		// 右摄像头参数路径及文件名


	vector< vector< Point3f > > object_points;				//特征点的世界坐标点坐标（mm）
	vector< vector< Point2f > > image_points;				//特征点的图像坐标点坐标（像素）
	int err = setup_calibration(object_points, image_points, board_width, board_height, num_imgs, square_size, imgs_directory, imgs_filename, extension);
	if (err == -1)
	{
		return -1;
	}
	printf("开始标定...\n");
	Mat cameraMatrix;
	Mat cameraDistcoeff;
	vector< Mat > rvecs, tvecs;
	int flag = 0;
	flag |= CV_CALIB_FIX_K4;
	flag |= CV_CALIB_FIX_K5;
	calibrateCamera(object_points, image_points, imgSize, cameraMatrix, cameraDistcoeff, rvecs, tvecs, flag);

	cv::FileStorage fs(out_file, cv::FileStorage::WRITE);
	fs << "cameraMatrix" << cameraMatrix;
	fs << "cameraDistcoeff" << cameraDistcoeff;
	fs << "board_width" << board_width;
	fs << "board_height" << board_height;
	fs << "square_size" << square_size;

	printf("标定完成！\n");

	return 0;
}

//读取图片并获取每张图片角点的两个坐标
int setup_calibration(	vector<vector<Point3f>> &object_points, vector<vector<Point2f>> &image_points, int board_width, int board_height, int num_imgs, float square_size, char* imgs_directory, char* imgs_filename, char* extension) 
{
	Mat img, gray;
	vector< Point2f > corners;
	vector< vector< Point2f > > left_img_points;
	Size board_size = Size(board_width, board_height);
	int board_n = board_width * board_height;

	for (int k = 1; k <= num_imgs; k++) 
	{
		char img_file[100];
		sprintf(img_file, "%s%s%d.%s", imgs_directory, imgs_filename, k, extension);
		img = imread(img_file, CV_LOAD_IMAGE_COLOR);
		if (img.empty())
		{
			printf("读入图片 %s 错误！\n", img_file);
			return -1;
		}

		cv::cvtColor(img, gray, CV_BGR2GRAY);

		bool found = false;
		found = cv::findChessboardCorners(img, board_size, corners,
			CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
		if (found)
		{
			cornerSubPix(gray, corners, cv::Size(5, 5), cv::Size(-1, -1),
				TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
			drawChessboardCorners(img, board_size, corners, found);
		}

		vector< Point3f > obj;
		for (int i = 0; i < board_height; i++)
			for (int j = 0; j < board_width; j++)
				obj.push_back(Point3f((float)j * square_size, (float)i * square_size, 0));

		if (found) 
		{
			cout << "图片 " << k << " 检测到角点!" << endl;
			image_points.push_back(corners);
			object_points.push_back(obj);
		}

		imshow("IMG", img);
		waitKey(10);
	}
	return 0;
}