#ifndef _FUNCTION_H_
#define _FUNCTION_H_
#define _USE_MATH_DEFINES
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/ml.hpp"


#include <map>

// include library using M_PI in math
#include <cmath>

using namespace std;
using namespace cv;

/**
 * Create Gaussian Kernel
 *
 * @param kernel - The kernel matrix destination.
 * @param ksize - The size of kernel with matrix ksize x ksize
 * @param sigma - The sigma of gaussian distribution.
 */
void createGaussianKernel(Mat& kernel, int ksize, float sigma);

Mat createLaplacianOfGaussian(int ksize, float sigma);

/**
* Conovle source matrix with kernel
*
* @param src - The matrix of source image
* @param dest - The matrix of destination image
* @param kernel - The kernel matrix
*/
void convolve(const Mat& src, Mat& dest, const Mat& kernel);


/**
* Apply GaussianBlur kernel to image
*
* @param src - The matrix of source image (gray scale)
* @param dest - The matrix of destination image
* @param kernel - The gaussian kernel
*
*/
void applyGaussianBlur(const Mat& src, Mat& dest, int ksize, float sigma);

Mat applyLOG(const Mat& src, int ksize, float sigma);


/**
* Convert matrix of image to new range that supports to show image with cv2::imshow()
* @param mat - The input matrix
* @param value - the value of scaling
* @return result - matrix that normalized
*/
Mat normalize(const Mat& mat, float value);

/**
* Compute Harris Responses
* @param k - Constant for computing Harris response (0.04 <= k <= 0.06)
* @param Ix - Gradient by horizontal X
* @param Iy - Gradient by vertical Y
* @return maxresponse - max response of matrix
*/
Mat computeHarrisResponses(float k, const Mat& Ix, const Mat& Iy, float& maxResponse);

/**
* Find corner points
*
* @param responseMat - response harris mat
* @param nmsThreshold - compute the threshold for non-maximum suppression - nmsThreshold * max(Harris Response)
* @param nmsWinSize - Non-maximum suppression window size.
* @param nmsWinSep - Non-maximum suppression window separation. Difference between two NMS window.
*
*/
vector<pair<int, int>> findCornerPoints(Mat& responseMat, float maxResponse, float nmsThreshold, int nmsWinize, int nmsWinSep);

/**
* Find local maximum point
*
* @param img - grayScaleImage - The matrix of source image (grayScaleImage scale)
* @param src - originalImage - The matrix of source image (original scale)
* @param cov - matrix of image at each scale.
* @param sigma - sigma value at each scale.
* @param maxValue - max pixel at each scale.
* @param no - number of scale (iteration).
* @param threshold - remove points under threshold*maxValue
* @param keypoints - key points of matrix
*
*/
void maximaDetection(Mat& img, Mat& src, vector<Mat> conv, vector<float> sigma, vector<float> maxValue, int no, float threshold, vector<pair<pair<int, int>, int>>& keypoints);

/**
* Harris Method
*
* @param originalImage - The matrix of source image (original scale)
* @param grayScaleImage - The matrix of source image (gray scale)
*
*/
void harrisMethod(Mat originalImage, Mat grayScaleImage);

/**
* Blob by LOG Method
*
* @param originalImage - The matrix of source image (original scale)
* @param grayScaleImage - The matrix of source image (gray scale)
*
*/
void blobMethod(Mat originalImage, Mat grayScaleImage);

/**
* Blob by DOG Method
*
* @param originalImage - The matrix of source image (original scale)
* @param grayScaleImage - The matrix of source image (gray scale)
*
*/
void DOGMethod(Mat originalImage, Mat grayScaleImage);

/**
* Detect corner by Harris
*
* @param img - The matrix of source image (gray scale)
* @param originalImage - The matrix of source image (original scale)
* @param ksize - the size of kernel with matrix ksize x ksize
* @param sigma - the sigma of gaussian distribution.
* @param nmsThresholdFraction - compute the threshold for non-maximum suppression - thresHoldNMS * max(Harris Response)
* @param nmsWinSize - Non-maximum suppression window size.
* @param nmsWinSep - Non-maximum suppression window separation. Difference between two NMS window.
*
*/
Mat detectHarris(Mat img, Mat originalImage, int ksize, float sigma, float nmsThreshold, float nmsWinSize, float nmsWinSep, vector<pair<pair<int, int>, int>>& keypoints);

/**
* Detect blob by LOG
*
* @param img - grayScaleImage - The matrix of source image (grayScaleImage scale)
* @param originalImage - originalImage - The matrix of source image (original scale)
* @param threshold - remove points under threshold*maxValue
* @param k - parametre of LOG method.
* @param s - start sigma value.
* @param no - number of scale (iteration).
* @param keypoints - key points of matrix
*
*/
Mat detectBlob(Mat img, Mat originalImage, float threshold, float k, float s, float no, vector<pair<pair<int, int>, int>>& keypoints);

/**
* Detect blob by DOG
*
* @param img - grayScaleImage - The matrix of source image (grayScaleImage scale)
* @param originalImage - originalImage - The matrix of source image (original scale)
* @param threshold - remove points under threshold*maxValue
* @param k - parametre of LOG method.
* @param s - start sigma value.
* @param no - number of scale (iteration).
* @param keypoints - key points of matrix
*
*/
Mat detectDOG(Mat img, Mat originalImage, float threshold, float k, float s, float no, vector<pair<pair<int, int>, int>>& keypoints);


/**
* SIFT Method
*
* @param originalImage1 - The matrix of source image 1 (original scale)
* @param originalImage2 - The matrix of source image 2 (original scale)
* @param grayScaleImage1 - The matrix of source image 1 (gray scale)
* @param grayScaleImage2 - The matrix of source image 2 (gray scale)
*
*/
void matchBySIFT(Mat originalImage1, Mat originalImage2, Mat grayScaleImage1, Mat grayScaleImage2, int detector);
#endif // !_FUNCTION_H_
