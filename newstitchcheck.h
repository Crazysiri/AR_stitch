#ifndef NEWSTITCHCHECK
#define NEWSTITCHCHECK
#define tmmin 1e-15

#include <opencv2/highgui/highgui.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/affine.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream>

#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/stitching/warpers.hpp"
#include<time.h>
using namespace cv;
using namespace std;


struct stitch_status
{
    int direction_status = 0; //方向状态
    /*
    -2   方向错误
    -1   异常
    0    没有拼接上
    1    能拼接但不够好，需要继续调整（屏幕上是红色的mask)
    2    可以很好拼接（屏幕上是绿色的mask）
    */
    Mat homo = (Mat_<double>(3, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    vector<Point2f> corner;
};

struct imagestatus
{
    int direction_status = 0;        //方向状态
    /*
    -1   异常
    0    没有拼接上
    1    拼接OK
    10   需要向左移动
    20   向右移动
    100  向上移动
    200  向下移动
    110  向左上移动
    120  向右上移动
    210  向左下移动
    220  向右下移动
    */
    double parallelslope = 0.0;        //基准图和待验图匹配点连线的水平斜率
    double verticalslope = 0.0;        //基准图和待验图匹配点连线的垂直斜率
    double alphaslope = 0.0;        //基准图和待验图匹配点连线的alpha斜率
    double betaslope = 0.0;            //基准图和待验图匹配点连线的beta斜率
    double density1 = 0;            //基准图匹配点的密度 图1
    double density2 = 0;            //待验图匹配点的密度 图2
    int matchnumber = 0;            //匹配点个数
    Mat homo = (Mat_<double>(3, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
};


struct featuredata
{
    vector<KeyPoint> keypoints;
    Mat descriptors;
    Mat image;
    cv::detail::ImageFeatures imageFeatures;
    Mat full_image;
};


struct homoandmask
{
    Mat homo = (Mat_<double>(3, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    vector<uchar> mask;
};


struct boxdata
{
    double xmin = 0;
    double ymin = 0;
    double xmax = 0;
    double ymax = 0;
};


int getfeaturedata(featuredata &result, Mat &image, int direction, double cutsize, double compression_ratio);
/*
// 获取特征信息
. @param image 输入图片
. @param direction 方向 0左 1右 2上 3下
. @param cutsize 切割尺寸
. @param compression_ratio 压缩比例
*/


int get_keypoints_and_descriptors(featuredata &result, Mat &image);
/*
用SURF提取特征点和特征图
*/



int check_image_v2(stitch_status &result, featuredata& basedata, Mat& image, int direction, double cutsize, double compression_ratio, int match_num1, int match_num2, double threshold1, double threshold2);
/*
// 检查图片状态
. @param basedata 输入基准图片信息
. @param image 输入待检测图片
. @param direction 检测区域 0左 1右 2上 3下
. @param cutsize 区域大小
. @param compression_ratio 压缩比例
. @match_num1：匹配点个数阈值1
. @match_num1：匹配点个数阈值2
. @threshold1：形变阈值1
. @threshold2：位置阈值2
*/


int checkimage(imagestatus &result, featuredata& basedata, Mat& image, int direction, double cutsize, double compression_ratio);
/*
// 检查图片状态
. @param basedata 输入基准图片信息
. @param image 输入待检测图片
. @param direction 检测区域 0左 1右 2上 3下
. @param cutsize 区域大小
. @param compression_ratio 压缩比例
*/


int computepoint(Point2f &dstpoint, Point2f &oripoint, Mat &homo);
/*
坐标变换
*/


int forback(vector<Point2f> &vp, vector<Point2f> &vp_);
/*
计算向前和向后
*/


int updown(Point2f &p, Point2f &p_, double d);
/*
计算上下
*/


int leftright(Point2f &p, Point2f &p_, double d);
/*
计算左右
*/


int rotate(Point2f &p1, Point2f &p2, double d);
/*
计算旋转
*/


int checkimagestatus(Mat& Image, Mat& homo, int direction, double cutsize = 0.5);
/*
// 检查图片状态
. @param Image 输入图片
. @param homo 输入变换矩阵
. @return int 状态信息 10左 20右 100上 200下
*/


int cutimage(Mat& result, Mat& image, int xmin, int ymin, int xmax, int ymax);
/*
// 切割图像
. @param image 输入图片
. @param xmin
. @param ymin
. @param xmax
. @param ymax
*/


int LoadImage(Mat& result, Mat& image, int direction, double cutsize, double compression_ratio);
/*
// 基准图切割
. @param image 输入图片
. @param direction 方向 0切割左边 1切割右边 2切割上边 3切割下边
. @param cutsize 切割尺寸
. @param compression_ratio 压缩比例
*/


int get_good_match_point(vector<DMatch> &result, Mat& descriptors1, Mat& descriptors2);
/*
// 根据匹配图获取匹配点
. @param descriptors1 特征图1
. @param descriptors2 特征图2
. @return vector<DMatch> 匹配点信息
*/


int gethomoandmask_v2(homoandmask &result, vector<KeyPoint> &keyPts1, vector<KeyPoint> &keyPts2, vector<DMatch> &GoodMatchePoints, int direction, Mat& image, double cutsize, int match_num);
/*
// 根据计算单应性矩阵获取匹配点
. @param keyPts1 特征图1
. @param keyPts2 特征图2
. @param GoodMatchePoints 匹配点关系信息
. @param direction 方向 0切割左边 1切割右边 2切割上边 3切割下边
. @param image 当前图片
. @cutsize 切割比例
. @match_num 匹配点个数阈值
. @return homoandmask
*/


int gethomoandmask_v3(homoandmask &result, vector<KeyPoint> &keyPts1, vector<KeyPoint> &keyPts2, vector<DMatch> &GoodMatchePoints, int direction, int h, int w, double cutsize, int match_num);
/*
// 根据计算单应性矩阵获取匹配点
. @param keyPts1 特征图1
. @param keyPts2 特征图2
. @param GoodMatchePoints 匹配点关系信息
. @param direction 方向 0切割左边 1切割右边 2切割上边 3切割下边
. @param h 当前图片的rows
. @param w 当前图片的cols
. @cutsize 切割比例
. @match_num 匹配点个数阈值
. @return homoandmask
*/


int gethomoandmask(homoandmask &result, vector<KeyPoint> &keyPts1, vector<KeyPoint> &keyPts2, vector<DMatch> &GoodMatchePoints);
/*
// 根据计算单应性矩阵获取匹配点
. @param keyPts1 特征图1
. @param keyPts2 特征图2
. @param GoodMatchePoints 匹配点关系信息
. @param direction 方向 0切割左边 1切割右边 2切割上边 3切割下边
. @param delta 坐标转换为切割前的坐标
. @return homoandmask
*/


int get_boxdata(boxdata &result, vector<Point2f>& points);
/*
// 计算所有匹配点的矩形区域
. @param points 匹配点
. @return boxdata result 边框信息
*/


void triangulation();

cv::Point2f calcWarpedPoint(
        const cv::Point2f& pt,
        InputArray K1,                // Camera K parameter
        InputArray R1,                // Camera R parameter
        InputArray K2,                // Camera K parameter
        InputArray R2,                // Camera R parameter
        cv::Ptr<cv::detail::RotationWarper> warper,  // The Rotation Warper
        Point_<int> corners1,
        Size_<int> sizes1,
        const std::vector<cv::Point> &corners2,
        const std::vector<cv::Size> &sizes2);




struct MyProjector
{
    void setCameraParams(InputArray K = Mat::eye(3, 3, CV_32F),
                         InputArray R = Mat::eye(3, 3, CV_32F),
                         InputArray T = Mat::zeros(3, 1, CV_32F));
    void mapForward(float x, float y, float &u, float &v);
    void mapBackward(float u, float v, float &x, float &y);

    float scale;
    float k[9];
    float rinv[9];
    float r_kinv[9];
    float k_rinv[9];
    float t[3];
};


int check_image_for_IOS(stitch_status &result, featuredata& basedata, Mat& image, int direction, double cutsize, double compression_ratio, int match_num1, int match_num2, double threshold1, double threshold2, int rows, int cols);


#endif
