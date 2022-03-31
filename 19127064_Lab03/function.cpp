#include "function.h"

void createGaussianKernel(Mat& kernel, int ksize, float sigma) {

	// initial kernel matrix
	Mat dest(ksize, ksize, CV_32F);
	int range = (ksize - 1) / 2;

	// initial the needed variable
	double sum = 0.0, r;
	float s = 2.0 * sigma * sigma;
	for (int x = -range; x <= range; ++x) {
		for (int y = -range; y <= range; ++y) {
			r = x * x + y * y;

			// Apply gaussian distribution
			dest.at<float>(x + range, y + range) = (exp(-(r / s))) / (M_PI * s);

			// Calculate sum to normalize
			sum += dest.at<float>(x + range, y + range);
		}
	}

	// Normalize the value of kernel
	for (int i = 0; i < ksize; ++i) {
		for (int j = 0; j < ksize; ++j) {
			dest.at<float>(i, j) /= sum;
		}
	}
	kernel = dest.clone();
}

pair<vector<float>, vector<int>> calcSigma(int no, float k, float s) {
	vector<float> sigma;
	vector<int> ksize;
	for (int i = 1; i <= no; i++) 
		sigma.push_back(s * pow(k, i));
	for (auto i : sigma)
		ksize.push_back(int(i * 6));
	for (int i = 0; i < ksize.size(); i++)
		if (ksize[i] % 2 == 0)
			ksize[i] += 1;
	return make_pair(sigma, ksize);
}

Mat createLaplacianOfGaussian(int ksize, float sigma) {

	// initial kernel matrix
	Mat dest(ksize, ksize, CV_32F);
	int range = (ksize - 1) / 2;
	// initial the needed variable

	float sum = 0.0, r;
	for (int x = -range; x <= range; ++x) {
		for (int y = -range; y <= range; ++y) {

			// Apply gaussian distribution
			float value1 = -((x * x + y * y) / (2.0 * sigma * sigma));
			float value2 = - 1.0 / (M_PI * sigma * sigma * sigma * sigma);
			dest.at<float>(x + range, y + range) = value2 * (1 + value1) * exp(value1);

			// Calculate sum to normalize
			sum += dest.at<float>(x + range, y + range);
		}
	}

	//Normalize the value of kernel
	for (int i = 0; i < ksize; ++i) {
		for (int j = 0; j < ksize; ++j) {
			dest.at<float>(i, j) /= sum;
		}
	}
	return dest;
}

void convolve(const Mat& src, Mat& dest, const Mat& kernel) {

	// initial destination matrix
	Mat result(src.rows, src.cols, CV_32F);

	int ksize = kernel.rows;

	// compute the center of matrix
	const int dx = ksize / 2;
	const int dy = ksize / 2;

	//loop height
	for (int i = 0; i < src.rows; ++i) {
		// loop width
		for (int j = 0; j < src.cols; ++j) {
			float temp = 0.0;
			for (int k = 0; k < ksize; ++k) {
				for (int l = 0; l < ksize; ++l) {
					int x = j - dx + l;
					int y = i - dy + k;

					// check position
					if (x >= 0 && x < src.cols && y >= 0 && y < src.rows) {
						if (kernel.type() == CV_32F && src.type() == CV_8U) {
							// reduce noise
							temp += src.at<uchar>(y, x) * kernel.at<float>(k, l);
						}
						else {
							temp += src.at<float>(y, x) * kernel.at<float>(k, l);
						}
					}
				}
			}

			//mapping to [0, 1]
			result.at<float>(i, j) = temp;
		}
	}
	dest = result.clone();
}


void applyGaussianBlur(const Mat& src, Mat& dest, int ksize, float sigma) {
	Mat kernel;

	// create gaussian kernel
	createGaussianKernel(kernel, ksize, sigma);
	convolve(src, dest, kernel);
}

Mat applyLOG(const Mat& src, int ksize, float sigma) {
	Mat kernel, dest;

	// create gaussian kernel
	kernel = createLaplacianOfGaussian(ksize, sigma);
	convolve(src, dest, kernel);
	return dest.clone();
}

Mat normalize(const Mat& mat, float value) {
	Mat result(mat.rows, mat.cols, CV_32F);
	for (int i = 0; i < mat.rows; ++i) {
		for (int j = 0; j < mat.cols; ++j) {
			result.at<float>(i, j) = mat.at<float>(i, j) * value;
		}
	}
	return result.clone();
}

Mat computeHarrisResponses(float k, const Mat& Ix, const Mat& Iy, float& maxResponse) {
	Mat result(Iy.rows, Ix.cols, CV_32F);
	int windowSize = 3;
	for (int i = 0; i < Iy.rows; ++i) {
		for (int j = 0; j < Iy.cols; ++j) {
			float a11 = 0, a12 = 0, a21 = 0, a22 = 0;
			for (int winRow = -windowSize/2; winRow <= windowSize/2; winRow++)
				for (int winCol = -windowSize/2; winCol <= windowSize/2; winCol++)
					if ((i + winRow) >= 0 && (i + winRow) < Ix.rows && ((j + winCol) >= 0 && (j + winCol) < Ix.cols)) {
						a11 += Ix.at<float>(i + winRow, j + winCol) * Ix.at<float>(i + winRow, j + winCol);
						a22 += Iy.at<float>(i + winRow, j + winCol) * Iy.at<float>(i + winRow, j + winCol);
						a21 += Ix.at<float>(i + winRow, j + winCol) * Iy.at<float>(i + winRow, j + winCol);
						a12 += Ix.at<float>(i + winRow, j + winCol) * Iy.at<float>(i + winRow, j + winCol);
					}
			float det = a11 * a22 - a12 * a21;
			float trace = a11 + a22;

			result.at<float>(i, j) = det - k * trace * trace;
			if (maxResponse < result.at<float>(i,j))
				maxResponse = result.at<float>(i, j);
		}
	}

	return result.clone();
}

vector<pair<int, int>> findCornerPoints(Mat& responseMat, float maxResponse, float thresholdNMS, int nmsWinSize, int nmsWinSep) {
	float thresholdVal = maxResponse * thresholdNMS, localMaxResponse;
	pair<int, int> responseCoordinates;
	vector<pair<int, int>> cornerCoordinates;

	for (int row = 0; row < responseMat.rows; row += nmsWinSep) {
		if (row + nmsWinSize >= responseMat.rows)
			continue;
		for (int col = 0; col < responseMat.cols; col += nmsWinSep) {
			if (col + nmsWinSize >= responseMat.cols)
				continue;
			localMaxResponse = FLT_MIN;
			for (int i = 0; i < nmsWinSize; i++)
				for (int j = 0; j < nmsWinSize; j++)
					if (localMaxResponse < responseMat.at<float>(row + i, col + j)) {
						localMaxResponse = responseMat.at<float>(row + i, col + j);
						responseCoordinates = make_pair(row + i, col + j);
					}
			if (localMaxResponse >= thresholdVal)
				cornerCoordinates.push_back(responseCoordinates);
			}
		}
	return cornerCoordinates;
}

void maximaDetection(Mat& img, Mat& src, vector<Mat> conv, vector<float> sigma, vector<float> maxValue, int no, float threshold, vector<pair<pair<int,int>, int>> &keypoints){
	for (int n = 0; n < conv.size(); n++) {
		int rad = int(sqrt(2) * sigma[n]);
		for (int row = 0; row < img.rows; ++row) {
			for (int col = 0; col < img.cols; ++col) {
				bool temp = true;
				for (int o = -1; o < 2; o++)
					for (int i = -1; i < 2; i++)
						for (int j = -1; j < 2; j++)
							if (row + i >= 0 && col + j >= 0 && row + i < img.rows && col + j < img.cols && n + o >= 0 && n + o < conv.size())
								if (conv[n].at<float>(row, col) > threshold*maxValue[n]) {
									if (conv[n].at<float>(row, col) < conv[n + o].at<float>(row + i, col + j))
										temp = false;
								}
								else temp = false;
				if (temp)
					if (row - rad > 0 && col - rad > 0 && row + rad < img.rows - 1 && col + rad < img.cols - 1) {
						keypoints.push_back(make_pair(make_pair(col, row), rad));
					}
			}
		}
	}
}

void substract(const Mat& mat1, const Mat& mat2, Mat& dest) {
	// check the shape of two matrices
	if (mat1.cols == mat2.cols && mat1.rows == mat2.rows) {
		// inital result matrix
		Mat result(mat1.rows, mat1.cols, CV_32F);
		for (int i = 0; i < mat1.rows; ++i) {
			for (int j = 0; j < mat1.cols; ++j) {
				result.at<float>(i, j) = mat1.at<float>(i, j) - mat2.at<float>(i, j);
			}
		}

		dest = result.clone();
	}
}

void multiply(const Mat& mat1, const Mat& mat2, Mat& dest) {
	// check the shape of two matrices
	if (mat1.cols == mat2.cols && mat1.rows == mat2.rows) {
		// inital result matrix
		Mat result(mat1.rows, mat1.cols, CV_32F);
		for (int i = 0; i < mat1.rows; ++i) {
			for (int j = 0; j < mat1.cols; ++j) {
				result.at<float>(i, j) = mat1.at<float>(i, j) * mat2.at<float>(i, j);
			}
		}

		dest = result.clone();
	}
}

float getMax(const Mat& mat1) {
	float max = mat1.at<float> (0,0);
	for (int i = 0; i < mat1.rows; ++i) {
		for (int j = 0; j < mat1.cols; ++j) {
			if (mat1.at<float>(i, j) > max)
				max = mat1.at<float>(i, j);
		}
	}
	//cout << max << endl;
	return max;
}

// Detect corner by Harris
Mat detectHarris(Mat img, Mat originalImage, int ksize, float sigma, float nmsThreshold, float nmsWinSize, float nmsWinSep, vector<pair<pair<int, int>, int>>& keypoints)
{
	Mat imageBlur, Ix, Iy, harrisResponses;
	try {
		// initial sobel filter
		float xFilters[3][3] = { {-1,0,1},{-2,0,2},{-1,0,1} };
		float yFilters[3][3] = { {-1,-2,-1},{0,0,0},{1,2,1} };
		Mat Kx(3, 3, CV_32F, xFilters);
		Mat Ky(3, 3, CV_32F, yFilters);


		// Apply Gaussian kernel to reduce noise 
		applyGaussianBlur(img, imageBlur, ksize, sigma);

		convolve(imageBlur, Ix, Kx);
		// Normailize the matrix of image to show
		Ix = normalize(Ix, 1.0 / 255);

		convolve(imageBlur, Iy, Ky);
		// Normailize the matrix of image to show
		Iy = normalize(Iy, 1.0 / 255);
		float maxResponse = FLT_MIN, k = 0.04;
		// Constant for computing Harris response (0.04 <= k <= 0.06)
		harrisResponses = computeHarrisResponses(k, Ix, Iy, maxResponse);
	
		vector<pair<int, int>> cornerCoordinates = findCornerPoints(harrisResponses, maxResponse, nmsThreshold, nmsWinSize, nmsWinSep);
		for (auto const& points : cornerCoordinates)
			keypoints.push_back(make_pair(make_pair(points.second, points.first), 3));
	}
	catch (Exception& e) {
		cout << e.msg << endl;
		return Mat();
	}
	return originalImage;
}

void harrisMethod(Mat originalImage, Mat grayScaleImage)
{
	Mat destX, destY, destXY;
	namedWindow("Harris", 1);

	// Default parameter
	int ksize = 3;
	float sigma = 1.0, nmsThreshold = 0.02, nmsWinSize = 3, nmsWinSep = 1;
	vector<pair<pair<int, int>, int>> keypoints;

	// Detect
	if (ksize % 2 != 0) {
		Mat res = detectHarris(grayScaleImage, originalImage, ksize, sigma, nmsThreshold, nmsWinSize, nmsWinSep, keypoints);
		for (auto const& points : keypoints)
			circle(originalImage, Point(points.first.first, points.first.second), 3, Scalar(0 ,255 , 0), 1);
		imshow("Harris", originalImage);
	}
	int iKey = waitKey(0);
}


//Detect blob LOG - Laplacian of the Gaussian
Mat detectBlob(Mat img, Mat originalImage, float threshold, float k, float s, float no, vector<pair<pair<int, int>, int>>& keypoints)
{
	try {
		pair<vector<float>, vector<int>> temp = calcSigma(no, k, s);
		vector<float> sigma = temp.first;
		vector<int> ksize = temp.second;
		vector<Mat> convOut;
		vector<float> maxLogValue;

		for (int i = 0; i < no; i++) {
			Mat temp = applyLOG(img, ksize[i], sigma[i]);
			temp = normalize(temp, 1.0 / 255);
			multiply(temp, temp, temp);
			maxLogValue.push_back(getMax(temp));
			convOut.push_back(temp);
		}
		maximaDetection(img, originalImage, convOut, sigma, maxLogValue, no, threshold, keypoints);
	}
	catch (Exception& e) {
		cout << e.msg << endl;
		return Mat();
	}
	return originalImage;
}

void blobMethod(Mat originalImage, Mat grayScaleImage)
{
	Mat destX, destY, destXY;
	//namedWindow("Blob", 1);

	// Default parameter
	int no = 8;
	//float threshold = 0.4, k = sqrt(2), s = 1/sqrt(2.5);
	float threshold = 0.4, k = sqrt(2), startSigma = 1;
	vector<pair<pair<int, int>, int>> keypoints;

	// Detect
	Mat res = detectBlob(grayScaleImage, originalImage, threshold, k, startSigma, no, keypoints);
	for (auto const& points : keypoints)
		circle(originalImage, Point(points.first.first, points.first.second), points.second, Scalar(0, 0, 255), 1);
	imshow("Blob", originalImage);
	int iKey = waitKey(0);
}


// Detect blob DoG - Difference of Gaussian
Mat detectDOG(Mat img, Mat originalImage, float threshold, float k, float s, float no, vector<pair<pair<int, int>, int>>& keypoints)
{
	try {
		pair<vector<float>, vector<int>> temp = calcSigma(no, k, s);
		vector<float> sigma = temp.first;
		vector<int> ksize = temp.second;
		vector<Mat> convOut;
		vector<float> maxLogValue;
		Mat preGau = applyLOG(img, ksize[0], sigma[0]), dogImg;
		preGau = normalize(preGau, 1.0/255);

		for (int i = 1; i < no; i++) {
			Mat currGau = applyLOG(img, ksize[i], sigma[i]);
			currGau = normalize(currGau, 1.0 / 255);
			substract(currGau, preGau, dogImg);
			maxLogValue.push_back(getMax(dogImg));
			//multiply(dogImg, dogImg, dogImg);
			convOut.push_back(dogImg);
			preGau = currGau.clone();
		}
		maximaDetection(img, originalImage, convOut, sigma, maxLogValue, no, threshold, keypoints);
	}
	catch (Exception& e) {
		cout << e.msg << endl;
		return Mat();
	}
	return originalImage;
}

void DOGMethod(Mat originalImage, Mat grayScaleImage)
{
	Mat destX, destY, destXY;
	namedWindow("DoG", 1);

	// Default parameter
	int no = 10;
	float threshold = 2, k = sqrt(2), startSigma = 1.0;
	vector<pair<pair<int, int>, int>> keypoints;

	// Detect
	Mat res = detectDOG(grayScaleImage, originalImage, threshold, k, startSigma, no, keypoints);
	for (auto const& points : keypoints)
		circle(originalImage, Point(points.first.first, points.first.second), points.second, Scalar(0, 255, 0), 1);
	imshow("DoG", originalImage);

	int iKey = waitKey(0);
}

// Match keypoints by SIFT
void matchBySIFT(Mat originalImage1, Mat originalImage2, Mat grayScaleImage1, Mat grayScaleImage2, int detector)
{
	Ptr<ml::KNearest> knn(ml::KNearest::create());
	Ptr<SiftDescriptorExtractor> extractor = SiftDescriptorExtractor::create();
	Mat descriptors1, descriptors2;
	vector<pair<pair<int, int>, int>> keypoints1_tmp, keypoints2_tmp;
	vector<KeyPoint> keypoints1, keypoints2;
	Mat res1, res2;

	//Harris
	if (detector == 1) {
		int ksize = 3;
		float sigma = 1.0, nmsThreshold = 0.02, nmsWinSize = 3, nmsWinSep = 1;
		// Detect
		if (ksize % 2 != 0) {
			res1 = detectHarris(grayScaleImage1, originalImage1, ksize, sigma, nmsThreshold, nmsWinSize, nmsWinSep, keypoints1_tmp);
			res2 = detectHarris(grayScaleImage2, originalImage2, ksize, sigma, nmsThreshold, nmsWinSize, nmsWinSep, keypoints2_tmp);
		}
	}

	// Blob
	if (detector == 2) {
		int no = 8;
		float threshold = 0.5, k = sqrt(2), startSigma = 1.0;
		// Detect
		res1 = detectBlob(grayScaleImage1, originalImage1, threshold, k, startSigma, no, keypoints1_tmp);
		res2 = detectBlob(grayScaleImage2, originalImage2, threshold, k, startSigma, no, keypoints2_tmp);
	}

	// DOG
	if (detector == 3) {
		int no = 8;
		float threshold = 0.7, k = sqrt(2), startSigma = 1.0;
		// Detect
		res1 = detectDOG(grayScaleImage1, originalImage1, threshold, k, startSigma, no, keypoints1_tmp);
		res2 = detectDOG(grayScaleImage2, originalImage2, threshold, k, startSigma, no, keypoints2_tmp);
	}

	for (auto i : keypoints1_tmp)
		keypoints1.push_back(KeyPoint(i.first.first, i.first.second, i.second));
	for (auto i : keypoints2_tmp)
		keypoints2.push_back(KeyPoint(i.first.first, i.first.second, i.second));

	extractor->compute(grayScaleImage1, keypoints1, descriptors1);
	extractor->compute(grayScaleImage2, keypoints2, descriptors2);
	cout << keypoints1.size() << endl;
	cout << keypoints2.size() << endl;
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	std::vector< std::vector<DMatch> > knn_matches;
	matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

	//-- Filter matches using the Lowe's ratio test
	const float ratio_thresh = 0.9f;
	std::vector<DMatch> good_matches;
	for (size_t i = 0; i < knn_matches.size(); i++)
	{
		if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
		{
			good_matches.push_back(knn_matches[i][0]);
		}
	}

	//-- Draw matches
	cout << good_matches.size();
	Mat img_matches;
	drawMatches(originalImage1, keypoints1, originalImage2, keypoints2, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::DEFAULT);

	//-- Show detected matches
	imshow("Good Matches", img_matches);
	int iKey = waitKey(0);
}
