#include <opencv2/opencv.hpp>
#include "inference.h"
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

int main() {
	std::string modelpath = "./models/best.onnx";
	std::string classpath = "./models/classes.txt";
	std::string videopath = "./video/testVideo.mp4";

	// gpu 미사용 시
	// Inference(modelpath, cv::Size(320, 320), classpath, false);

	// gpu 사용 시
	Inference inf(modelpath, cv::Size(640, 640), classpath, true);

	cv::VideoCapture cap(videopath);

	if (!cap.isOpened())
	{
		std::cerr << "Error: Could not open video file." << std::endl;
		return -1;
	}

	cv::Mat frame;
	while (cap.read(frame))
	{
		std::vector<Detection> output = inf.runInference(frame);

		for ( auto& detection : output)
		{
			cv::rectangle(frame, detection.box, detection.color, 2);
			std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
			cv::putText(frame, classString, cv::Point(detection.box.x, detection.box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2);
		}

		cv::imshow("Inference", frame);
		if (cv::waitKey(30) >= 0) break;
	}

	cap.release();
	return 0;
}
