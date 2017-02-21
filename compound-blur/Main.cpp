#include <opencv2\highgui.hpp>
#include <opencv2\opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

Mat mat1, mat2, mat3;
int *pixels1;
string w1 = "w1", w2 = "w2";
int dotx = 300, doty = 200;
int distance1, distance2;
int maxBlur = 31, w, h;
Mat rot_mat;
Mat lines;
bool translated = false;
int centerx, centery, angle, steps;
Mat gradientData;
float startRadius, radiusFactor;

int mul_table[] = {
	512, 512, 456, 512, 328, 456, 335, 512, 405, 328, 271, 456, 388, 335, 292, 512,
	454, 405, 364, 328, 298, 271, 496, 456, 420, 388, 360, 335, 312, 292, 273, 512,
	482, 454, 428, 405, 383, 364, 345, 328, 312, 298, 284, 271, 259, 496, 475, 456,
	437, 420, 404, 388, 374, 360, 347, 335, 323, 312, 302, 292, 282, 273, 265, 512,
	497, 482, 468, 454, 441, 428, 417, 405, 394, 383, 373, 364, 354, 345, 337, 328,
	320, 312, 305, 298, 291, 284, 278, 271, 265, 259, 507, 496, 485, 475, 465, 456,
	446, 437, 428, 420, 412, 404, 396, 388, 381, 374, 367, 360, 354, 347, 341, 335,
	329, 323, 318, 312, 307, 302, 297, 292, 287, 282, 278, 273, 269, 265, 261, 512,
	505, 497, 489, 482, 475, 468, 461, 454, 447, 441, 435, 428, 422, 417, 411, 405,
	399, 394, 389, 383, 378, 373, 368, 364, 359, 354, 350, 345, 341, 337, 332, 328,
	324, 320, 316, 312, 309, 305, 301, 298, 294, 291, 287, 284, 281, 278, 274, 271,
	268, 265, 262, 259, 257, 507, 501, 496, 491, 485, 480, 475, 470, 465, 460, 456,
	451, 446, 442, 437, 433, 428, 424, 420, 416, 412, 408, 404, 400, 396, 392, 388,
	385, 381, 377, 374, 370, 367, 363, 360, 357, 354, 350, 347, 344, 341, 338, 335,
	332, 329, 326, 323, 320, 318, 315, 312, 310, 307, 304, 302, 299, 297, 294, 292,
	289, 287, 285, 282, 280, 278, 275, 273, 271, 269, 267, 265, 263, 261, 259 };
int shg_table[] = {
	9, 11, 12, 13, 13, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 17,
	17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19,
	19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20,
	20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21,
	21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
	21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22,
	22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22,
	22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 23,
	23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
	23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
	23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
	23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
	24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
	24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
	24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
	24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24 };

class BlurStack {
public:
	int r, g, b, a;
	BlurStack *next;
	BlurStack() {
		r = g = b = a = 0;
		next = NULL;
	}
};
void matToArray(Mat m, int *a) {
	int ii = 0;
	for (int y = 0; y < m.rows; y++) {
		for (int x = 0; x < m.cols; x++) {
			Vec3b color = m.at<Vec3b>(y, x);
			a[ii] = (int)color[0];
			a[ii + 1] = (int)color[1];
			a[ii + 2] = (int)color[2];
			ii += 4;
		}
	}
}
void arrayToMat(int* a, Mat &m) {
	int ii = 0;
	for (int y = 0; y < m.rows; y++) {
		for (int x = 0; x < m.cols; x++) {
			Vec3b color(a[ii], a[ii + 1], a[ii + 2]);
			m.at<Vec3b>(y, x) = color;
			ii += 4;
		}
	}
}
void mat8UC1ToArray(Mat m, int *a) {
	int ii = 0;
	for (int y = 0; y < m.rows; y++) {
		for (int x = 0; x < m.cols; x++) {
			int value = m.at<uchar>(y, x);
			a[ii] = value;
			a[ii + 1] = value;
			a[ii + 2] = value;
			a[ii + 3] = 255;
			ii += 4;
		}
	}
}
Mat getLinearGradientMap(int width, int height, int centerX, int centerY, int d1, int d2, int a)
{
	int w = 2 * sqrt(width*width + height*height);
	int h = w;// (d1 + d2) * 2;
	int out = h / 2 - (d1 + d2);
	Mat m1(h, w, CV_8UC1, Scalar(0, 0, 0));
	Mat m2(height, width, CV_8UC1);
	float stepR = (float)255 / d2;
	for (int r = 0; r < out; r++) {
		m1.row(r).setTo(255);
	}
	for (int r = 0; r < d2; r++) {
		m1.row(r + out).setTo(255 - r*stepR);
	}
	int temp = 2 * d1 + d2 + out;
	for (int r = 0; r < d2; r++) {
		m1.row(r + temp).setTo(r*stepR);
	}
	for (int r = h - out - 1; r < h; r++) {
		m1.row(r).setTo(255);
	}
	//m1.at<uchar>(h / 2, w / 2) = 0;
	Mat rotateMat = getRotationMatrix2D(Point(w / 2, h / 2), a, 1.0);
	Mat newrow(1, 3, CV_64F, (double)0);
	newrow.at<double>(0, 2) = 1;
	rotateMat.push_back(newrow);

	Mat finalMat;

	Mat translateMat = (Mat_<double>(2, 3) << 1, 0, centerX - w / 2, 0, 1, centerY - h / 2);

	translateMat.push_back(newrow);
	finalMat = translateMat*rotateMat;
	finalMat.pop_back();
	warpAffine(m1, m2, finalMat, m2.size());


	return m2;
}
void renderCompoundBlurRGBA(Mat src, Mat &dst, Mat radiusData, int width, int height, float radius, float increaseFactor, int blurLevels) {
	int x, y, i, p, yp, yi, yw, r_sum, g_sum, b_sum, a_sum,
		r_out_sum, g_out_sum, b_out_sum, a_out_sum,
		r_in_sum, g_in_sum, b_in_sum, a_in_sum,
		pa, pr, pg, pb, rbs;
	int wh = width * height;
	int wh4 = wh << 2;

	int* imagePixels = new int[wh4];
	int* radiusPixels = new int[wh4];
	matToArray(src, imagePixels);
	mat8UC1ToArray(radiusData, imagePixels);

	int* pixels = new int[wh4];
	for (int i = 0; i < wh4; i++)
	{
		pixels[i] = imagePixels[i];
	}

	int currentIndex = 0;
	int steps = blurLevels;
	blurLevels -= 1;

	while (steps-- >= 0)
	{
		int iradius = (radius + 0.5)/* | 0*/;
		if (iradius == 0) continue;
		if (iradius > 256) iradius = 256;

		float div = iradius + iradius + 1;
		int w4 = width << 2;
		int widthMinus1 = width - 1;
		int heightMinus1 = height - 1;
		float radiusPlus1 = iradius + 1;
		int sumFactor = radiusPlus1 * (radiusPlus1 + 1) / 2;

		BlurStack *stackStart = new BlurStack();
		BlurStack *stack = stackStart;
		BlurStack *stackEnd = NULL;
		for (i = 1; i < div; i++)
		{
			stack->next = new BlurStack();
			stack = stack->next;
			if (i == radiusPlus1) stackEnd = stack;
		}
		stack->next = stackStart;
		BlurStack *stackIn = NULL;
		BlurStack *stackOut = NULL;

		yw = yi = 0;

		int mul_sum = mul_table[iradius];
		int shg_sum = shg_table[iradius];

		for (y = 0; y < height; y++)
		{
			r_in_sum = g_in_sum = b_in_sum = a_in_sum = r_sum = g_sum = b_sum = a_sum = 0;

			r_out_sum = radiusPlus1 * (pr = pixels[yi]);
			g_out_sum = radiusPlus1 * (pg = pixels[yi + 1]);
			b_out_sum = radiusPlus1 * (pb = pixels[yi + 2]);
			a_out_sum = radiusPlus1 * (pa = pixels[yi + 3]);

			r_sum += sumFactor * pr;
			g_sum += sumFactor * pg;
			b_sum += sumFactor * pb;
			a_sum += sumFactor * pa;

			stack = stackStart;

			for (i = 0; i < radiusPlus1; i++)
			{
				stack->r = pr;
				stack->g = pg;
				stack->b = pb;
				stack->a = pa;
				stack = stack->next;
			}

			for (i = 1; i < radiusPlus1; i++)
			{
				p = yi + ((widthMinus1 < i ? widthMinus1 : i) << 2);
				r_sum += (stack->r = (pr = pixels[p])) * (rbs = radiusPlus1 - i);
				g_sum += (stack->g = (pg = pixels[p + 1])) * rbs;
				b_sum += (stack->b = (pb = pixels[p + 2])) * rbs;
				a_sum += (stack->a = (pa = pixels[p + 3])) * rbs;

				r_in_sum += pr;
				g_in_sum += pg;
				b_in_sum += pb;
				a_in_sum += pa;

				stack = stack->next;
			}


			stackIn = stackStart;
			stackOut = stackEnd;
			for (x = 0; x < width; x++)
			{
				pixels[yi + 3] = pa = (a_sum * mul_sum) >> shg_sum;
				if (pa != 0)
				{
					pa = 255 / pa;
					pixels[yi] = ((r_sum * mul_sum) >> shg_sum) * pa;
					pixels[yi + 1] = ((g_sum * mul_sum) >> shg_sum) * pa;
					pixels[yi + 2] = ((b_sum * mul_sum) >> shg_sum) * pa;
				}
				else {
					pixels[yi] = pixels[yi + 1] = pixels[yi + 2] = 0;
				}

				r_sum -= r_out_sum;
				g_sum -= g_out_sum;
				b_sum -= b_out_sum;
				a_sum -= a_out_sum;

				r_out_sum -= stackIn->r;
				g_out_sum -= stackIn->g;
				b_out_sum -= stackIn->b;
				a_out_sum -= stackIn->a;

				p = (yw + ((p = x + radiusPlus1) < widthMinus1 ? p : widthMinus1)) << 2;

				r_in_sum += (stackIn->r = pixels[p]);
				g_in_sum += (stackIn->g = pixels[p + 1]);
				b_in_sum += (stackIn->b = pixels[p + 2]);
				a_in_sum += (stackIn->a = pixels[p + 3]);

				r_sum += r_in_sum;
				g_sum += g_in_sum;
				b_sum += b_in_sum;
				a_sum += a_in_sum;

				stackIn = stackIn->next;

				r_out_sum += (pr = stackOut->r);
				g_out_sum += (pg = stackOut->g);
				b_out_sum += (pb = stackOut->b);
				a_out_sum += (pa = stackOut->a);

				r_in_sum -= pr;
				g_in_sum -= pg;
				b_in_sum -= pb;
				a_in_sum -= pa;

				stackOut = stackOut->next;

				yi += 4;
			}
			yw += width;
		}


		for (x = 0; x < width; x++)
		{
			g_in_sum = b_in_sum = a_in_sum = r_in_sum = g_sum = b_sum = a_sum = r_sum = 0;

			yi = x << 2;
			r_out_sum = radiusPlus1 * (pr = pixels[yi]);
			g_out_sum = radiusPlus1 * (pg = pixels[yi + 1]);
			b_out_sum = radiusPlus1 * (pb = pixels[yi + 2]);
			a_out_sum = radiusPlus1 * (pa = pixels[yi + 3]);

			r_sum += sumFactor * pr;
			g_sum += sumFactor * pg;
			b_sum += sumFactor * pb;
			a_sum += sumFactor * pa;

			stack = stackStart;

			for (i = 0; i < radiusPlus1; i++)
			{
				stack->r = pr;
				stack->g = pg;
				stack->b = pb;
				stack->a = pa;
				stack = stack->next;
			}

			yp = width;

			for (i = 1; i < radiusPlus1; i++)
			{
				yi = (yp + x) << 2;

				r_sum += (stack->r = (pr = pixels[yi])) * (rbs = radiusPlus1 - i);
				g_sum += (stack->g = (pg = pixels[yi + 1])) * rbs;
				b_sum += (stack->b = (pb = pixels[yi + 2])) * rbs;
				a_sum += (stack->a = (pa = pixels[yi + 3])) * rbs;

				r_in_sum += pr;
				g_in_sum += pg;
				b_in_sum += pb;
				a_in_sum += pa;

				stack = stack->next;

				if (i < heightMinus1)
				{
					yp += width;
				}
			}

			yi = x;
			stackIn = stackStart;
			stackOut = stackEnd;
			for (y = 0; y < height; y++)
			{
				p = yi << 2;
				pixels[p + 3] = pa = (a_sum * mul_sum) >> shg_sum;
				if (pa > 0)
				{
					pa = 255 / pa;
					pixels[p] = ((r_sum * mul_sum) >> shg_sum) * pa;
					pixels[p + 1] = ((g_sum * mul_sum) >> shg_sum) * pa;
					pixels[p + 2] = ((b_sum * mul_sum) >> shg_sum) * pa;
				}
				else {
					pixels[p] = pixels[p + 1] = pixels[p + 2] = 0;
				}

				r_sum -= r_out_sum;
				g_sum -= g_out_sum;
				b_sum -= b_out_sum;
				a_sum -= a_out_sum;

				r_out_sum -= stackIn->r;
				g_out_sum -= stackIn->g;
				b_out_sum -= stackIn->b;
				a_out_sum -= stackIn->a;

				p = (x + (((p = y + radiusPlus1) < heightMinus1 ? p : heightMinus1) * width)) << 2;

				r_sum += (r_in_sum += (stackIn->r = pixels[p]));
				g_sum += (g_in_sum += (stackIn->g = pixels[p + 1]));
				b_sum += (b_in_sum += (stackIn->b = pixels[p + 2]));
				a_sum += (a_in_sum += (stackIn->a = pixels[p + 3]));

				stackIn = stackIn->next;

				r_out_sum += (pr = stackOut->r);
				g_out_sum += (pg = stackOut->g);
				b_out_sum += (pb = stackOut->b);
				a_out_sum += (pa = stackOut->a);

				r_in_sum -= pr;
				g_in_sum -= pg;
				b_in_sum -= pb;
				a_in_sum -= pa;

				stackOut = stackOut->next;

				yi += width;
			}
		}

		radius *= increaseFactor;

		for (i = wh; --i > -1; )
		{
			int idx = i << 2;
			int lookupValue = (radiusPixels[idx + 2] & 0xff) / 255.0 * blurLevels;
			int index = lookupValue | 0;

			if (index == currentIndex)
			{
				int blend = 256.0 * (lookupValue - (lookupValue | 0));
				int iblend = 256 - blend;

				imagePixels[idx] = (imagePixels[idx] * iblend + pixels[idx] * blend) >> 8;
				imagePixels[idx + 1] = (imagePixels[idx + 1] * iblend + pixels[idx + 1] * blend) >> 8;
				imagePixels[idx + 2] = (imagePixels[idx + 2] * iblend + pixels[idx + 2] * blend) >> 8;
				imagePixels[idx + 3] = (imagePixels[idx + 3] * iblend + pixels[idx + 3] * blend) >> 8;

			}
			else if (index == currentIndex + 1)
			{
				imagePixels[idx] = pixels[idx];
				imagePixels[idx + 1] = pixels[idx + 1];
				imagePixels[idx + 2] = pixels[idx + 2];
				imagePixels[idx + 3] = pixels[idx + 3];
			}
		}
		currentIndex++;
	}
	////////////////////////////
	arrayToMat(imagePixels, dst);
}
void compoundBlurCanvasRGBA(Mat src, Mat &dst, int topx, int topy, int w, int h, Mat radiusData, float minRadius, float increaseFactor, int blurLevels) {
	renderCompoundBlurRGBA(src, dst, radiusData, w, h, minRadius, increaseFactor, blurLevels);
	//context.putImageData(imageData, top_x, top_y);
}
void compoundBlurCanvasRGB(Mat src, Mat &dst, int topx, int topy, int width, int height, Mat radiusData, float radius, float increaseFactor, int blurLevels) {
	int x, y, i, p, yp, yi, yw, r_sum, g_sum, b_sum,
		r_out_sum, g_out_sum, b_out_sum,
		r_in_sum, g_in_sum, b_in_sum,
		pr, pg, pb, rbs;
	int wh = width * height;
	int wh4 = wh << 2;
	int *pixels = new int[wh4];

	int* imagePixels = new int[wh4];
	int* radiusPixels = new int[wh4];
	matToArray(src, imagePixels);
	mat8UC1ToArray(radiusData, radiusPixels);

	for (int i = 0; i < wh4; i++)
	{
		pixels[i] = imagePixels[i];
	}

	int currentIndex = 0;
	int steps = blurLevels;
	blurLevels -= 1;

	while (steps-- >= 0)
	{
		int iradius = (radius + 0.5)/* | 0*/;
		if (iradius == 0) continue;
		if (iradius > 256) iradius = 256;
		int div = iradius + iradius + 1;
		int w4 = width << 2;
		int widthMinus1 = width - 1;
		int heightMinus1 = height - 1;
		int radiusPlus1 = iradius + 1;
		int sumFactor = radiusPlus1 * (radiusPlus1 + 1) / 2;

		BlurStack *stackStart = new BlurStack();
		BlurStack *stack = stackStart;
		BlurStack *stackEnd = NULL;
		for (i = 1; i < div; i++)
		{
			stack = stack->next = new BlurStack();
			if (i == radiusPlus1) stackEnd = stack;
		}
		stack->next = stackStart;
		BlurStack *stackIn = NULL;
		BlurStack *stackOut = NULL;

		yw = yi = 0;

		int mul_sum = mul_table[iradius];
		int shg_sum = shg_table[iradius];

		for (y = 0; y < height; y++)
		{
			r_in_sum = g_in_sum = b_in_sum = r_sum = g_sum = b_sum = 0;

			r_out_sum = radiusPlus1 * (pr = pixels[yi]);
			g_out_sum = radiusPlus1 * (pg = pixels[yi + 1]);
			b_out_sum = radiusPlus1 * (pb = pixels[yi + 2]);

			r_sum += sumFactor * pr;
			g_sum += sumFactor * pg;
			b_sum += sumFactor * pb;

			stack = stackStart;

			for (i = 0; i < radiusPlus1; i++)
			{
				stack->r = pr;
				stack->g = pg;
				stack->b = pb;
				stack = stack->next;
			}

			for (i = 1; i < radiusPlus1; i++)
			{
				p = yi + ((widthMinus1 < i ? widthMinus1 : i) << 2);
				r_sum += (stack->r = (pr = pixels[p])) * (rbs = radiusPlus1 - i);
				g_sum += (stack->g = (pg = pixels[p + 1])) * rbs;
				b_sum += (stack->b = (pb = pixels[p + 2])) * rbs;

				r_in_sum += pr;
				g_in_sum += pg;
				b_in_sum += pb;

				stack = stack->next;
			}


			stackIn = stackStart;
			stackOut = stackEnd;
			for (x = 0; x < width; x++)
			{
				pixels[yi] = (r_sum * mul_sum) >> shg_sum;
				pixels[yi + 1] = (g_sum * mul_sum) >> shg_sum;
				pixels[yi + 2] = (b_sum * mul_sum) >> shg_sum;

				r_sum -= r_out_sum;
				g_sum -= g_out_sum;
				b_sum -= b_out_sum;

				r_out_sum -= stackIn->r;
				g_out_sum -= stackIn->g;
				b_out_sum -= stackIn->b;

				p = (yw + ((p = x + radiusPlus1) < widthMinus1 ? p : widthMinus1)) << 2;

				r_in_sum += (stackIn->r = pixels[p]);
				g_in_sum += (stackIn->g = pixels[p + 1]);
				b_in_sum += (stackIn->b = pixels[p + 2]);

				r_sum += r_in_sum;
				g_sum += g_in_sum;
				b_sum += b_in_sum;

				stackIn = stackIn->next;

				r_out_sum += (pr = stackOut->r);
				g_out_sum += (pg = stackOut->g);
				b_out_sum += (pb = stackOut->b);

				r_in_sum -= pr;
				g_in_sum -= pg;
				b_in_sum -= pb;

				stackOut = stackOut->next;

				yi += 4;
			}
			yw += width;
		}


		for (x = 0; x < width; x++)
		{
			g_in_sum = b_in_sum = r_in_sum = g_sum = b_sum = r_sum = 0;

			yi = x << 2;
			r_out_sum = radiusPlus1 * (pr = pixels[yi]);
			g_out_sum = radiusPlus1 * (pg = pixels[yi + 1]);
			b_out_sum = radiusPlus1 * (pb = pixels[yi + 2]);

			r_sum += sumFactor * pr;
			g_sum += sumFactor * pg;
			b_sum += sumFactor * pb;

			stack = stackStart;

			for (i = 0; i < radiusPlus1; i++)
			{
				stack->r = pr;
				stack->g = pg;
				stack->b = pb;
				stack = stack->next;
			}

			yp = width;

			for (i = 1; i < radiusPlus1; i++)
			{
				yi = (yp + x) << 2;

				r_sum += (stack->r = (pr = pixels[yi])) * (rbs = radiusPlus1 - i);
				g_sum += (stack->g = (pg = pixels[yi + 1])) * rbs;
				b_sum += (stack->b = (pb = pixels[yi + 2])) * rbs;

				r_in_sum += pr;
				g_in_sum += pg;
				b_in_sum += pb;

				stack = stack->next;

				if (i < heightMinus1)
				{
					yp += width;
				}
			}

			yi = x;
			stackIn = stackStart;
			stackOut = stackEnd;
			for (y = 0; y < height; y++)
			{
				p = yi << 2;
				pixels[p] = (r_sum * mul_sum) >> shg_sum;
				pixels[p + 1] = (g_sum * mul_sum) >> shg_sum;
				pixels[p + 2] = (b_sum * mul_sum) >> shg_sum;

				r_sum -= r_out_sum;
				g_sum -= g_out_sum;
				b_sum -= b_out_sum;

				r_out_sum -= stackIn->r;
				g_out_sum -= stackIn->g;
				b_out_sum -= stackIn->b;

				p = (x + (((p = y + radiusPlus1) < heightMinus1 ? p : heightMinus1) * width)) << 2;

				r_sum += (r_in_sum += (stackIn->r = pixels[p]));
				g_sum += (g_in_sum += (stackIn->g = pixels[p + 1]));
				b_sum += (b_in_sum += (stackIn->b = pixels[p + 2]));

				stackIn = stackIn->next;

				r_out_sum += (pr = stackOut->r);
				g_out_sum += (pg = stackOut->g);
				b_out_sum += (pb = stackOut->b);

				r_in_sum -= pr;
				g_in_sum -= pg;
				b_in_sum -= pb;

				stackOut = stackOut->next;

				yi += width;
			}
		}

		radius *= increaseFactor;
		for (i = wh; --i > -1; )
		{
			int idx = i << 2;
			float lookupValue = (radiusPixels[idx + 2] & 0xff) / 255.0 * blurLevels;
			int index = lookupValue /*| 0*/;
			if (index == currentIndex)
			{
				float blend = 256.0 * (lookupValue - (int)lookupValue);
				float iblend = 256 - blend;

				imagePixels[idx] = (int)(imagePixels[idx] * iblend + pixels[idx] * blend) >> 8;
				imagePixels[idx + 1] = (int)(imagePixels[idx + 1] * iblend + pixels[idx + 1] * blend) >> 8;
				imagePixels[idx + 2] = (int)(imagePixels[idx + 2] * iblend + pixels[idx + 2] * blend) >> 8;
			}
			else if (index == currentIndex + 1)
			{
				imagePixels[idx] = pixels[idx];
				imagePixels[idx + 1] = pixels[idx + 1];
				imagePixels[idx + 2] = pixels[idx + 2];
			}
		}
		currentIndex++;
	}
	////////////////////
	arrayToMat(imagePixels, dst);
}
void compoundBlurImage(Mat src, Mat &dst, Mat radiusData, float minRadius, float increaseFactor, int blurLevels, bool blurAlphaChannel)
{
	int w = src.cols;
	int h = src.rows;

	CV_Assert(minRadius > 0 && increaseFactor != 0);
	if (src.type() == CV_8UC4)
		compoundBlurCanvasRGBA(src, dst, 0, 0, w, h, radiusData, minRadius, increaseFactor, blurLevels);
	else if (src.type() == CV_8UC3)
		compoundBlurCanvasRGB(src, dst, 0, 0, w, h, radiusData, minRadius, increaseFactor, blurLevels);
}
float calculateStartRadius(int blurRadius) {
	steps = 2;
	radiusFactor = 1.5;
	float divider = radiusFactor;
	for (int i = 1; i < steps; i++)
	{
		divider += pow(radiusFactor, i + 1);
	}
	return blurRadius / divider;
}
void onChangeAngle(int value, void* ptr) {
	angle = value;
	gradientData = getLinearGradientMap(w, h, centerx, centery, distance1, distance2, angle);
	imshow(w1, gradientData);
	compoundBlurCanvasRGB(mat1, mat2, 0, 0, w, h, gradientData, startRadius, radiusFactor, steps);
	imshow(w2, mat2);
}
void onChangeX(int value, void* ptr) {
	centerx = value;
	gradientData = getLinearGradientMap(w, h, centerx, centery, distance1, distance2, angle);
	imshow(w1, gradientData);
	compoundBlurCanvasRGB(mat1, mat2, 0, 0, w, h, gradientData, startRadius, radiusFactor, steps);
	imshow(w2, mat2);
}
void onChangeY(int value, void* ptr) {
	centery = value;
	gradientData = getLinearGradientMap(w, h, centerx, centery, distance1, distance2, angle);
	imshow(w1, gradientData);
	compoundBlurCanvasRGB(mat1, mat2, 0, 0, w, h, gradientData, startRadius, radiusFactor, steps);
	imshow(w2, mat2);
}
void onChangeRadius(int value, void* ptr) {
	startRadius = calculateStartRadius(value);
	compoundBlurCanvasRGB(mat1, mat2, 0, 0, w, h, gradientData, startRadius, radiusFactor, steps);
	imshow(w2, mat2);
}
void onChangeDistance1(int value, void* ptr) {
	distance1 = value;
	gradientData = getLinearGradientMap(w, h, centerx, centery, distance1, distance2, angle);
	compoundBlurCanvasRGB(mat1, mat2, 0, 0, w, h, gradientData, startRadius, radiusFactor, steps);
	imshow(w1, gradientData);
	imshow(w2, mat2);
}
void onChangeDistance2(int value, void* ptr) {
	distance2 = value;
	gradientData = getLinearGradientMap(w, h, centerx, centery, distance1, distance2, angle);
	imshow(w1, gradientData);
	compoundBlurCanvasRGB(mat1, mat2, 0, 0, w, h, gradientData, startRadius, radiusFactor, steps);
	imshow(w2, mat2);
}
int main() {
	mat1 = imread("s1.jpg", IMREAD_COLOR);
	mat1.copyTo(mat2);
	w = mat1.cols;
	h = mat1.rows;
	centerx = 200; centery = 200; angle = 0, distance1 = 80, distance2 = 100;

	int blurRadius = 70;
	startRadius = calculateStartRadius(blurRadius);

	gradientData = getLinearGradientMap(w, h, centerx, centery, distance1, distance2, angle);
	compoundBlurCanvasRGB(mat1, mat2, 0, 0, w, h, gradientData, startRadius, radiusFactor, steps);

	namedWindow(w1, WINDOW_AUTOSIZE);
	namedWindow(w2, WINDOW_AUTOSIZE);
	createTrackbar("rot", w1, 0, 180, onChangeAngle);
	createTrackbar("x", w1, 0, w, onChangeX);
	createTrackbar("y", w1, 0, h, onChangeY);
	createTrackbar("radius", w2, 0, 100, onChangeRadius);
	createTrackbar("distance1", w2, 0, 200, onChangeDistance1);
	createTrackbar("distance2", w2, 0, 200, onChangeDistance2);
	imshow(w1, getLinearGradientMap(w, h, centerx, centery, distance1, distance2, angle));
	imshow(w2, mat2);
	waitKey(0);
}