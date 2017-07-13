// ����У�����������ͼ��任�������ƽ�棬���У�����ͼƬ

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
	char* leftimg_directory = "..//..//imgs//leftImgs//";		// ������ͷͼ��·��
	char* rightimg_directory = "..//..//imgs//rightImgs//";		// ������ͷͼ��·��
	char* leftimg_filename = "left";							// ������ͷͼ����
	char* rightimg_filename = "right";							// ������ͷͼ����
	char* extension = "jpg";									// ͼ���ʽ
	char* calib_file = "..//..//parms//calibParms.xml";			// ����У�������ļ�
	char* rectifiedImg_directory = "..//..//result//rectifiedImg//";// ͼ��У�����·��
	char* leftout_directory = "..//..//result//left//";			// ������ͷͼ��У�����·��
	char* rightout_directory = "..//..//result//right//";		// ������ͷͼ��У�����·��
	char* rectifiedImg_filename = "rectifiedImg";				// У�����ͼ���ļ���
	char* leftout_filename = "leftOut";							// ������ͷͼ��У������ļ���
	char* rightout_filename = "rightOut";						// ������ͷͼ��У������ļ���
	int num_imgs = 17;
	Size imgSize = Size(640,480);

	Mat R1, R2, P1, P2, Q;
	Mat cameraMatrixL, cameraDistcoeffL, cameraMatrixR, cameraDistcoeffR, R, T;

	cv::FileStorage fs(calib_file, cv::FileStorage::READ);
	if (!fs.isOpened())
	{
		printf("�򿪲����ļ�ʧ��!\n");
		return -1;
	}
	fs["cameraMatrixL"] >> cameraMatrixL;
	fs["cameraDistcoeffL"] >> cameraDistcoeffL;
	fs["cameraMatrixR"] >> cameraMatrixR;
	fs["cameraDistcoeffR"] >> cameraDistcoeffR;
	fs["R"] >> R;
	fs["T"] >> T;

	fs["R1"] >> R1;//�����һ�������3x3�����任(��ת����)
	fs["R2"] >> R2;
	fs["P1"] >> P1;//�ڵ�һ̨������µ�����ϵͳ(��������)��� 3x4 ��ͶӰ����
	fs["P2"] >> P2;
	fs["Q"] >> Q;  //����Ӳ�ӳ�����

	cout << "������" << endl << endl;
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

	cout << "�����������..." << endl;

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
			printf("����ͼ�����\n");
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


		// ��ʾ����

		Mat rectifiedImg;
		double sf = 1.0;
		int w = cvRound(imgSize.width * sf), h = cvRound(imgSize.height * sf);
		rectifiedImg.create(h, w * 2, CV_8UC3);

		// ��������ͼ��
		Mat rectifiedImgPartL = rectifiedImg(Rect(w * 0, 0, w, h)); 
		Mat rectifiedImgPartR = rectifiedImg(Rect(w, 0, w, h));
		resize(imgU1, rectifiedImgPartL, rectifiedImgPartL.size(), 0, 0, INTER_AREA);
		resize(imgU2, rectifiedImgPartR, rectifiedImgPartR.size(), 0, 0, INTER_LINEAR);
		Rect vroiL(cvRound(validROIL.x * sf), cvRound(validROIL.y * sf), 
			cvRound(validROIL.width * sf), cvRound(validROIL.height * sf)); // ������Ч����
		Rect vroiR(cvRound(validROIR.x * sf), cvRound(validROIR.y * sf), 
			cvRound(validROIR.width * sf), cvRound(validROIR.height * sf));
		rectangle(rectifiedImgPartL, vroiL, Scalar(0, 0, 255), 3, 8);		// ����Ч�����Ͼ���  
		rectangle(rectifiedImgPartR, vroiR, Scalar(0, 0, 255), 3, 8); 

		// ���϶�Ӧ������
		for (int i = 0; i < rectifiedImg.rows; i += 16)
			line(rectifiedImg, Point(0, i), Point(rectifiedImg.cols, i), Scalar(0, 255, 0), 1, 8);
		imshow("rectified", rectifiedImg);

		// ����ͼ��
		char rectifiedImg_file[100];
		sprintf(rectifiedImg_file, "%s%s%d.%s", rectifiedImg_directory, rectifiedImg_filename, k, extension);
		imwrite(rectifiedImg_file, rectifiedImg);

		waitKey(0);
	}
	
	return 0;
}