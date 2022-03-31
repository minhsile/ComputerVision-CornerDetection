#include "function.h"

int main(int argc, char** argv) {
	cv::CommandLineParser parser(argc, argv,
		"{image1 |Images\\1.png|Input image's path}"
		"{image2 |DataSet\\training_images\\15_3.jpg|Input image's path}"
		"{method |Harris|Choose method [Harris, Blob, DOG, SIFT] to detect corner of image }"
		"{detector |1|1 - Harris, 2 - Blob, 3 - DOG}"
	);
	// Show help's commandline 
	parser.about("\n~~This program detect corner of image~~\n[Press ESC to exit program]");
	parser.printMessage();

	// Get image's path
	String imageSrc = parser.get<String>("image1");
	try {
		Mat originalImage1 = imread(imageSrc, IMREAD_COLOR), grayScaleImage1;
		if (originalImage1.empty()) {
			cout << "Path doesn't exist";
			return 0;
		}
		// Resize image to 512x512
		resize(originalImage1, originalImage1, Size(512, 512));

		// Convert color's image to grayscale
		if (originalImage1.channels() == 3)
			cvtColor(originalImage1, grayScaleImage1, COLOR_BGR2GRAY);
		else grayScaleImage1 = originalImage1;

		// Get method option
		String method = parser.get<String>("method");

		// Harris method
		if (method == "Harris") {
			imshow("Orginal image", originalImage1);
			harrisMethod(originalImage1, grayScaleImage1);
		}

		// Blob
		if (method == "Blob") {
			imshow("Orginal image", originalImage1);
			blobMethod(originalImage1, grayScaleImage1);
		}

		// Blob by DoG
		if (method == "DoG") {
			imshow("Orginal image", originalImage1);
			DOGMethod(originalImage1, grayScaleImage1);
		}

		if (method == "SIFT") {
			String imageSrc2 = parser.get<String>("image2");
			int detector = parser.get<int>("detector");
			Mat originalImage2 = imread(imageSrc2, IMREAD_COLOR), grayScaleImage2;
			if (originalImage2.empty()) {
				cout << "Path doesn't exist";
				return 0;
			}
			// Resize image to 512x512
			resize(originalImage2, originalImage2, Size(512, 512));

			// Convert color's image to grayscale
			if (originalImage2.channels() == 3)
				cvtColor(originalImage2, grayScaleImage2, COLOR_BGR2GRAY);
			else grayScaleImage2 = originalImage2;
			//imshow("Orginal image", originalImage2);
			matchBySIFT(originalImage1, originalImage2, grayScaleImage1, grayScaleImage2, detector);
		}
	}
	catch (Exception& e) {
		cout << e.msg;
		return 0;
	}
	return 0;
}