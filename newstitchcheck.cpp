#include "newstitchcheck.h"
#include "exception"
#include <stdlib.h>
#define GOODMATCHNUMBER 20
#define n_max 600;
double work_megapix = 0.6;
double seam_megapix = 0.1;
double compose_megapix = -1;
float conf_thresh = 1.f;
float match_conf = 0.5f;
float blend_strength = 5;
double work_scale = 1, seam_scale = 1, compose_scale = 1;
bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;
float seam_work_aspect = 1.0f;
#ifdef __IPHONE_OS_VERSION_MAX_ALLOWED //使用一个iOS肯定有的宏定义 来判断是否是iOS 或者 安卓
Ptr<cv::xfeatures2d::SiftFeatureDetector> finder = cv::xfeatures2d::SiftFeatureDetector::create(1000);
#define WIDTH_MAX 800
#else

//! feature keypoint descrpetor finder
Ptr<Feature2D> finder = xfeatures2d::SURF::create(2000, 4,3, false, false);

#define WIDTH_MAX 600
#endif
// Ptr<Feature2D>  finder = ORB::create();
class MyPoint
{
public:
    double x, y, z;
    MyPoint(double x=0.0, double y=0.0, double z=1.0)
    {
        this->x = x;
        this->y = y;
        this->z = z;
    }
    void calculatenewpoint(Mat& H)
    {
        Mat point_old = (Mat_<double>(3, 1) << this->x, this->y, this->z);
        Mat point_new = H * point_old;
        point_new /= point_new.at<double>(2, 0);
        this->x = point_new.at<double>(0, 0);
        this->y = point_new.at<double>(1, 0);
        this->z = point_new.at<double>(2, 0);
    }
};


class Corner
{
public:
    MyPoint ltop;
    MyPoint lbottom;
    MyPoint rtop;
    MyPoint rbottom;

    void calculatefromimage(Mat& img)
    {
        int rows = img.rows;
        int cols = img.cols;
        this->ltop.x = 0.0;
        this->ltop.y = 0.0;
        this->lbottom.x = 0.0;
        this->lbottom.y = float(rows);
        this->rtop.x = float(cols);
        this->rtop.y = 0.0;
        this->rbottom.x = float(cols);
        this->rbottom.y = float(rows);
    }

    void calculatefromhomo(Mat& H)
    {
        this->ltop.calculatenewpoint(H);
        this->lbottom.calculatenewpoint(H);
        this->rtop.calculatenewpoint(H);
        this->rbottom.calculatenewpoint(H);
    }

};


int calculatecorners(Corner& c, Mat& img, Mat& H) {
    c.calculatefromimage(img);
    c.calculatefromhomo(H);
    return 0;
}
//! calculate currect point
inline
void MyProjector::mapForward(float x, float y, float &u, float &v)
{
    float x_ = r_kinv[0] * x + r_kinv[1] * y + r_kinv[2];
    float y_ = r_kinv[3] * x + r_kinv[4] * y + r_kinv[5];
    float z_ = r_kinv[6] * x + r_kinv[7] * y + r_kinv[8];

    x_ = t[0] + x_ / z_ * (1 - t[2]);
    y_ = t[1] + y_ / z_ * (1 - t[2]);

    u = scale * x_;
    v = scale * y_;
}

inline
void MyProjector::mapBackward(float u, float v, float &x, float &y)
{
    u = u / scale - t[0];
    v = v / scale - t[1];

    float z;
    x = k_rinv[0] * u + k_rinv[1] * v + k_rinv[2] * (1 - t[2]);
    y = k_rinv[3] * u + k_rinv[4] * v + k_rinv[5] * (1 - t[2]);
    z = k_rinv[6] * u + k_rinv[7] * v + k_rinv[8] * (1 - t[2]);

    x /= z;
    y /= z;
}

void MyProjector::setCameraParams(InputArray _K, InputArray _R, InputArray _T)
{
    Mat K = _K.getMat(), R = _R.getMat(), T = _T.getMat();

    CV_Assert(K.size() == Size(3, 3) && K.type() == CV_32F);
    CV_Assert(R.size() == Size(3, 3) && R.type() == CV_32F);
    CV_Assert((T.size() == Size(1, 3) || T.size() == Size(3, 1)) && T.type() == CV_32F);

    Mat_<float> K_(K);
    k[0] = K_(0,0); k[1] = K_(0,1); k[2] = K_(0,2);
    k[3] = K_(1,0); k[4] = K_(1,1); k[5] = K_(1,2);
    k[6] = K_(2,0); k[7] = K_(2,1); k[8] = K_(2,2);

    Mat_<float> Rinv = R.t();
    rinv[0] = Rinv(0,0); rinv[1] = Rinv(0,1); rinv[2] = Rinv(0,2);
    rinv[3] = Rinv(1,0); rinv[4] = Rinv(1,1); rinv[5] = Rinv(1,2);
    rinv[6] = Rinv(2,0); rinv[7] = Rinv(2,1); rinv[8] = Rinv(2,2);

    Mat_<float> R_Kinv = R * K.inv();
    r_kinv[0] = R_Kinv(0,0); r_kinv[1] = R_Kinv(0,1); r_kinv[2] = R_Kinv(0,2);
    r_kinv[3] = R_Kinv(1,0); r_kinv[4] = R_Kinv(1,1); r_kinv[5] = R_Kinv(1,2);
    r_kinv[6] = R_Kinv(2,0); r_kinv[7] = R_Kinv(2,1); r_kinv[8] = R_Kinv(2,2);

    Mat_<float> K_Rinv = K * Rinv;
    k_rinv[0] = K_Rinv(0,0); k_rinv[1] = K_Rinv(0,1); k_rinv[2] = K_Rinv(0,2);
    k_rinv[3] = K_Rinv(1,0); k_rinv[4] = K_Rinv(1,1); k_rinv[5] = K_Rinv(1,2);
    k_rinv[6] = K_Rinv(2,0); k_rinv[7] = K_Rinv(2,1); k_rinv[8] = K_Rinv(2,2);

    Mat_<float> T_(T.reshape(0, 3));
    t[0] = T_(0,0); t[1] = T_(1,0); t[2] = T_(2,0);
}

Point2f warpPoint(const Point2f &pt, InputArray K, InputArray R, float scale)
{
    MyProjector projector_;
    projector_.scale = scale;
    float tz[] = {0.f, 0.f, 0.f};
    Mat_<float> T(3, 1, tz);
    projector_.setCameraParams(K, R, T);

    Point2f uv;
    projector_.mapForward(pt.x, pt.y, uv.x, uv.y);
    return uv;
}

Point2f dewarpPoint(const Point2f &pt, InputArray K, InputArray R, float scale)
{
    MyProjector projector_;
    projector_.scale = scale;
    float tz[] = {0.f, 0.f, 0.f};
    Mat_<float> T(3, 1, tz);
    projector_.setCameraParams(K, R, T);

    Point2f uv;
    projector_.mapBackward(pt.x, pt.y, uv.x, uv.y);
    return uv;
}

cv::Point2f calcWarpedPoint(
        const cv::Point2f& pt,
        InputArray K0, InputArray R0,
        InputArray K1, InputArray R1,
        Ptr<cv::detail::RotationWarper> warper)
{
    cv::Point2f  dst = warper->warpPoint(pt, K0, R0);
    float scale = warper->getScale();
    cv::Point2f final_pt = dewarpPoint(dst, K1, R1, scale);
    return final_pt;
}



int gethomoandmask_v3(homoandmask &result, vector<KeyPoint> &keyPts1, vector<KeyPoint> &keyPts2, vector<DMatch> &GoodMatchePoints, int direction, int h_, int w_, double cutsize, int match_num)
{
    result.mask.clear();
    vector<Point2f> imagePoints1, imagePoints2;
    double ratio;
    double delta = 0;
    if (direction == 0 or direction == 1) {
        ratio = (double)(h_) / n_max;
        int w = (int)(w_ / ratio);
        delta = w * (1 - cutsize);
    } else {
        ratio = (double)(w_) / n_max;
        int h = (int)(h_ / ratio);
        delta = h * (1 - cutsize);
    }
    if (GoodMatchePoints.size() < match_num) { return 1; }
    for (auto & GoodMatchePoint : GoodMatchePoints) {
        Point2f pt1 = keyPts1[GoodMatchePoint.queryIdx].pt;
        Point2f pt2 = keyPts2[GoodMatchePoint.trainIdx].pt;
        if (direction == 0) {
            pt1.x += delta;
        } else if (direction == 1) {
            pt2.x += delta;
        } else if (direction == 2) {
            pt1.y += delta;
        } else if (direction == 3) {
            pt2.y += delta;
        }
        pt1.x = pt1.x * ratio;
        pt1.y = pt1.y * ratio;
        pt2.x = pt2.x * ratio;
        pt2.y = pt2.y * ratio;

        imagePoints1.push_back(pt1);
        imagePoints2.push_back(pt2);
    }
    if (imagePoints1.size() != imagePoints2.size() && imagePoints1.size() < match_num && imagePoints2.size() < match_num) {
        return 1;
    }
    vector<uchar> mask;
    Mat homo = (Mat_<double>(2, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0);

    Mat K(Matx33d(
            2759.48, 0, 1520.69,
            0, 2764.16, 1006.81,
            0, 0, 1
    ));

    double focal_length = 0.5*(K.at<double>(0) + K.at<double>(4));
    Point2d principle_point(K.at<double>(2), K.at<double>(5));

    Mat E = findEssentialMat(imagePoints1, imagePoints2, focal_length, principle_point, RANSAC, 0.999, 1.0, mask);
    if (E.empty()) return 0;

//    homo = findFundamentalMat(imagePoints1, imagePoints2, FM_RANSAC, 3, 0.99);
//    homo =estimateAffine2D(Mat(imagePoints1),Mat(imagePoints2), mask,RANSAC,8,2000);
    homo = findHomography(Mat(imagePoints1), Mat(imagePoints2), RHO, 7.0, mask,3000);
//    Mat homo1 = getAffineTransform(imagePoints1, imagePoints2);
    cout<<homo <<"\n";
    cout<<E <<"\n";
    result.homo = homo;
    if (!homo.empty() && homo.rows == 3 && homo.cols == 3) {
        result.homo = homo;
    }
//    cout<<"\n"<<homo<<"\n";
//    Mat homo_c1 = (Mat_<double>(3, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
//    Mat R1 = homo_c1(Range(0,3), Range(0,3));
//    Mat R2 = homo(Range(0,3), Range(0,3));
//
//    Mat R_2to1 = R1*R2.t();

    result.mask = mask;
    return 0;
}

//! the new alogrithm is using camera matrix K and rotation matrix R to calculate the mapping area
int check_image_v2(stitch_status &result, featuredata& basedata, Mat& image, int direction, double cutsize, double compression_ratio, int match_num1, int match_num2, double threshold1, double threshold2)
{
    try{
        //! init the status, 0 is can not stitching
        result.direction_status = 0;
        //    work_scale = float(image.rows)/float(800);
        //! 手机屏幕分辨率是1920*1080
        Size target_size;
        target_size.height = 1920;
        target_size.width = 1080;

        if (image.channels() == 3) {
            cvtColor(image, image, COLOR_RGB2GRAY);
        }

        resize(image, image, target_size);
        cv::Size full_img_size;

        full_img_size.height = image.rows;
        full_img_size.width = image.cols;

        vector<Size> full_img_sizes(2);
        full_img_sizes[0] = full_img_size;
        full_img_sizes[1] = full_img_size;

        //! cut image based one the cut percent
//        int cols = 1080;
//        int rows = 1920;
//        switch (direction) {
//            case 0:
//                cutimage(image, image, 0, 0, (int)(cols * cutsize), rows);
//                break;
//            case 1:
//                cutimage(image, image, (int)(cols * (1 - cutsize)), 0, cols, rows);
//                break;
//            case 2:
//                cutimage(image, image, 0, 0, cols, (int)(rows * cutsize));
//                break;
//            case 3:
//                cutimage(image, image, 0, (int)(rows * (1 - cutsize)), cols, rows);
//                break;
//            default:
//                break;
//        }



        Mat full_v2_image = image.clone();

        //! resize percent
        work_scale = min(1.0, sqrt(work_megapix * 1e6 / image.size().area()));
        work_scale = float(WIDTH_MAX)/float(image.rows);
        resize(image, image, Size(), work_scale, work_scale, 5);

        //! an empty keypoint and decriptor calculator
        cv::detail::ImageFeatures image2Feature;
        seam_scale = 1.0;
        seam_work_aspect = seam_scale / work_scale;
        cv::detail::computeImageFeatures(finder, image, image2Feature);

        //! cause we calculate the keypoints based on the cut image
        //! so we need to map the keypoint to whole image
//        for(int i = 0; i < image2Feature.keypoints.size();i++){
//
//            //! 0 cut right
//            //! 1 cut left, so we need change the cooridinate add the cut size to the x
//            //! 2 cut below
//            //! 3 cut up, so need to add the cut size to the y
//            switch (direction) {
//                case 0:
//                    break;
//                case 1:
//                    image2Feature.keypoints[i].pt = Point2f(image2Feature.keypoints[i].pt.x + ((int)(cols * (1 - cutsize)))*work_scale,image2Feature.keypoints[i].pt.y);
//                    break;
//                case 2:
//                    break;
//                case 3:
//                    image2Feature.keypoints[i].pt = Point2f(image2Feature.keypoints[i].pt.x,(image2Feature.keypoints[i].pt.y)+((int)(rows * (1 - cutsize)))*work_scale);
//                    break;
//                default:
//                    break;
//            }
//        }
        //! create a feature vector to store the keypoints and features
        vector<cv::detail::ImageFeatures> features;
        features.push_back(basedata.imageFeatures);
        features.push_back(image2Feature);

        //! match creator match image features
        vector<cv::detail::MatchesInfo> pairwise_matches;
        Ptr<cv::detail::FeaturesMatcher>  matcher = makePtr<cv::detail::BestOf2NearestMatcher>(false, match_conf);
        (*matcher)(features, pairwise_matches);
        matcher->collectGarbage();

        //! resize
        vector<Mat> images(2);
        resize(basedata.image, images[0], Size(), seam_scale, seam_scale, 5);
        resize(image, images[1], Size(), seam_scale, seam_scale, 5);
        //! (4) 剔除外点，保留最确信的大成分
        // Leave only images we are sure are from the same panorama
        vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
        vector<Mat> img_subset;
        cout<< "num of inliers is : "<<pairwise_matches[1].num_inliers<<"\n";

//        if(pairwise_matches[1].num_inliers < 10){
//
//        }
//
//        //! 匹配点的数量小于50
//        if(pairwise_matches[1].num_inliers < 50){
//            result.direction_status = 1;
//        }
        vector<Size> full_img_sizes_subset;
        for (size_t i = 0; i < indices.size(); ++i)
        {

            img_subset.push_back(images[indices[i]]);
            full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
        }
        if(img_subset.size() < 2){
            return 0;
        }
        images = img_subset;
        full_img_sizes = full_img_sizes_subset;

        // Check if we still have enough images
        int num_images = static_cast<int>(img_subset.size());
        if (num_images < 2)
        {
            result.direction_status = 0;
            std::cout << "Need more images\n";
            result.direction_status = result.direction_status +  pairwise_matches[1].num_inliers*10;
            return -1;
        }

        //!(5) 估计 homography
        Ptr<cv::detail::Estimator> estimator = makePtr<cv::detail::HomographyBasedEstimator>();
        vector<cv::detail::CameraParams> cameras;
        if (!(*estimator)(features, pairwise_matches, cameras))
        {
            cout << "Homography estimation failed.\n";
            result.direction_status = result.direction_status +  pairwise_matches[1].num_inliers*10;
            return 0;
        }


        for (size_t i = 0; i < cameras.size(); ++i)
        {
            Mat R;
            cameras[i].R.convertTo(R, CV_32F);
            cameras[i].R = R;
            std::cout << "\nInitial camera intrinsics #" << indices[i] + 1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R << std::endl;
        }

        cout<< "The camera matrix "<<cameras[0].K()<<"\n";
        cout<< "The camera matrix "<<cameras[0].R<<"\n";
        cout<< "The camera matrix "<<cameras[1].R<<"\n";

        //(6) 创建约束调整器
        Ptr<detail::BundleAdjusterBase> adjuster = makePtr<detail::BundleAdjusterRay>();
        adjuster->setConfThresh(conf_thresh);
        Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
        refine_mask(0, 0) = 1;
        refine_mask(0, 1) = 1;
        refine_mask(0, 2) = 1;
        refine_mask(1, 1) = 1;
        refine_mask(1, 2) = 1;
        adjuster->setRefinementMask(refine_mask);
        if (!(*adjuster)(features, pairwise_matches, cameras))
        {

            cout << "Camera parameters adjusting failed.\n";
            result.direction_status = result.direction_status +  pairwise_matches[1].num_inliers*10;
            return -1;
        }
        cout<< "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n";
        cout<< "The camera matrix "<<cameras[0].K()<<"\n";
        cout<< "The camera matrix "<<cameras[0].R<<"\n";
        cout<< "The camera matrix "<<cameras[1].R<<"\n";
        cout<< "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n";


        // Find median focal length
        vector<double> focals;
        for (size_t i = 0; i < cameras.size(); ++i)
        {
            focals.push_back(cameras[i].focal);
        }

        //! calculate the warp image scale
        sort(focals.begin(), focals.end());
        float warped_image_scale;
        if (focals.size() % 2 == 1)
            warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
        else
            warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;


        std::cout << "\nWarping images (auxiliary)... \n";



        vector<UMat> images_warped(num_images);
        vector<Size> sizes(num_images);

        //! Warp images and their masks
//        Ptr<WarperCreator> warper_creator = makePtr<cv::CylindricalWarper>();
        Ptr<WarperCreator> warper_creator = makePtr<cv::PlaneWarper>();
        if (!warper_creator)
        {
            cout << "Can't create the warper \n";
            result.direction_status = result.direction_status +  pairwise_matches[1].num_inliers*10;
            return 1;
        }

        //! Create RotationWarper
        Ptr<cv::detail::RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));


        images.clear();
        images_warped.clear();


        Mat img_warped, img_warped_s;
        Mat dilated_mask, seam_mask, mask, mask_warped;
        double compose_work_aspect = 1;
        is_compose_scale_set = false;
        for (int img_idx = 0; img_idx < num_images; ++img_idx)
        {
            // Read image and resize it if necessary
            Mat full_img = full_v2_image.clone();
            if (!is_compose_scale_set)
            {
                is_compose_scale_set = true;
                compose_work_aspect = compose_scale / work_scale;

                // Update warped image scale
                warped_image_scale *= static_cast<float>(compose_work_aspect);
                warper = warper_creator->create(warped_image_scale);

                // Update corners and sizes
                for (int i = 0; i < num_images; ++i)
                {
                    cameras[i].focal *= compose_work_aspect;
                    cameras[i].ppx *= compose_work_aspect;
                    cameras[i].ppy *= compose_work_aspect;
                }
            }


            full_img.release();
            img_warped.convertTo(img_warped_s, CV_16S);
            img_warped.release();

            mask.release();

        }



        Mat K0;
        cameras[0].K().convertTo(K0, CV_32F);
        Mat R0;
        cameras[0].R.convertTo(R0, CV_32F);


        Mat K1;
        cameras[1].K().convertTo(K1, CV_32F);
        Mat R1;
        cameras[1].R.convertTo(R1, CV_32F);



        vector<pair<int, int> > img_sizes;
        for (int idx = 0; idx < 2; ++idx) {

            img_sizes.push_back(make_pair(basedata.full_image.cols, basedata.full_image.rows));
        }

        cv::Point2f p0 = cv::Point2f(0,0);
        cv::Point2f p1 = cv::Point2f(1080,0);
        cv::Point2f p2 = cv::Point2f(1080,1920);
        cv::Point2f p3 = cv::Point2f(0,1920);



        cv::Point p0_ = calcWarpedPoint(p0, K0, R0, K1, R1, warper);
        cv::Point p1_ = calcWarpedPoint(p1, K0, R0, K1, R1, warper);
        cv::Point p2_ = calcWarpedPoint(p2, K0, R0, K1, R1, warper);
        cv::Point p3_ = calcWarpedPoint(p3, K0, R0, K1, R1, warper);


        bool write_image = false;
        if (write_image){
            std::cout << "***************************************" << std::endl;
            Point root_points[1][4];
            root_points[0][0] = p0_;
            root_points[0][1] = p1_;
            root_points[0][2] = p2_;
            root_points[0][3] = p3_;

            std::cout << p0_ << "\n";
            std::cout << p1_ << "\n";
            std::cout << p2_ << "\n";
            std::cout << p3_ << "\n";


            std::cout << p0 << "\n";
            std::cout << p1 << "\n";
            std::cout << p2 << "\n";
            std::cout << p3 << "\n";

            const Point* ppt[1] = {root_points[0]};
            int npt[] = {4};


            polylines(full_v2_image,  ppt, npt, 1, 1, Scalar(0,255,0),4,8,0);

            std::cout << full_v2_image.cols << "\n";
            std::cout << full_v2_image.rows << "\n";
            std::cout << "***************************************" << std::endl;

            std::cout << "\nCheck `result.png`, `result_mask.png` and `result2.png`!\n";
            imwrite("/home/baihao/jpg/resultxuboabab.jpg", full_v2_image);
        }


        result.corner = vector<Point2f>({p0_, p1_,
                                         p2_, p3_});

        result.direction_status = 2;
//        if(pairwise_matches[1].num_inliers < 50){
//            result.direction_status = 1 + pairwise_matches[1].num_inliers*10;
//        }else{
//            result.direction_status = 2 + pairwise_matches[1].num_inliers*10;
//        }
        result.homo = refine_mask;
        return 0;
    } catch (...) {
        result.direction_status = -1;
        return 1;
    }

    return 1;



}

//! detecte feature and descrptor using Imagefeature detetor the algorithm logical is not change
int getfeaturedata(featuredata &result, Mat &image, int direction, double cutsize, double compression_ratio)
{
    Mat *image_ = new Mat();
    try
    {
        if (image.empty())
        {
            (*image_).release();
            delete image_;
            return 0;
        }
        Size target_size;
        target_size.height = 1920;
        target_size.width = 1080;
//        work_scale = float(800)/float(image.rows);
        if (image.channels() == 3) {
            cvtColor(image, image, COLOR_RGB2GRAY);
        }
        resize(image, image, target_size);

        int cols = 1080;
        int rows = 1920;
//        switch (direction) {
//            case 0:
//                cutimage(image, image, 0, 0, (int)(cols * cutsize), rows);
//                break;
//            case 1:
//                cutimage(image, image, (int)(cols * (1 - cutsize)), 0, cols, rows);
//                break;
//            case 2:
//                cutimage(image, image, 0, 0, cols, (int)(rows * cutsize));
//                break;
//            case 3:
//                cutimage(image, image, 0, (int)(rows * (1 - cutsize)), cols, rows);
//                break;
//            default:
//                break;
//        }
//        imwrite("/home/baihao/jpg/check.jpg",image);

//        work_scale = min(1.0, sqrt(work_megapix * 1e6 / image.size().area()));
        work_scale = float(WIDTH_MAX)/float(image.rows);
        result.full_image = image;
        resize(image, result.image, Size(), work_scale, work_scale, 5);


//        seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / image.size().area()));
        seam_scale = 1.0;
        seam_work_aspect = seam_scale / work_scale;
        cv::detail::computeImageFeatures(finder, result.image, result.imageFeatures);
//        for(int i = 0; i < result.imageFeatures.keypoints.size();i++){
//
//
//            switch (direction) {
//                case 0:
////                    cutimage(image, image, 0, 0, (int)(cols * cutsize), rows);
////                    result.imageFeatures.keypoints[i].pt = Point2f(result.imageFeatures.keypoints[i].pt.x,result.imageFeatures.keypoints[i].pt.y);
//                    break;
//                case 1:
////                    cutimage(image, image, (int)(cols * (1 - cutsize)), 0, cols, rows);
//                    result.imageFeatures.keypoints[i].pt = Point2f(result.imageFeatures.keypoints[i].pt.x + ((int)(cols * (1 - cutsize)))*work_scale,result.imageFeatures.keypoints[i].pt.y);
//                    break;
//                case 2:
////                    cutimage(image, image, 0, 0, cols, (int)(rows * cutsize));
//                    break;
//                case 3:
////                    cutimage(image, image, 0, (int)(rows * (1 - cutsize)), cols, rows);
//                    result.imageFeatures.keypoints[i].pt = Point2f(result.imageFeatures.keypoints[i].pt.x,(result.imageFeatures.keypoints[i].pt.y)+((rows * (1 - cutsize)))*work_scale);
//                    break;
//                default:
//                    break;
//            }
//        }
//        resize(result.image, result.image, Size(), seam_scale, seam_scale, 5);
//
//        resize(image,image,target_size);
//        LoadImage(*image_, image, direction, cutsize, compression_ratio);
//        get_keypoints_and_descriptors(result, *image_);
//        result.image = image;

        (*image_).release();
        delete image_;
        return 1;
    }
    catch (...)
    {
        (*image_).release();
        delete image_;
        return 0;
    }
}

//! original feature and descrpitor calculator
int get_keypoints_and_descriptors(featuredata &result, Mat &image)
{
    try {
        Mat *M = new Mat();
        image.copyTo(*M);
        if (image.channels() == 3) {
            cvtColor(*M, *M, COLOR_RGB2GRAY);
        }
        vector<KeyPoint> *keypoints = new vector<KeyPoint>;
        Mat *descriptors = new Mat();

//        Ptr<Feature2D> f2d = xfeatures2d::SURF::create();
#ifdef __IPHONE_OS_VERSION_MAX_ALLOWED //使用一个iOS肯定有的宏定义 来判断是否是iOS 或者 安卓
        Ptr<cv::xfeatures2d::SiftFeatureDetector> f2d = cv::xfeatures2d::SiftFeatureDetector::create(1000);
#else
        Ptr<Feature2D> f2d = xfeatures2d::SURF::create(1000);
#endif
//        Ptr<AKAZE> f2d = AKAZE::create();
//        Ptr<cv::xfeatures2d::SiftFeatureDetector> f2d = cv::xfeatures2d::SiftFeatureDetector::create();
        int step = 10;
//        vector<KeyPoint> kps;
        for (int i=step; i<image.rows-step; i+=step)
        {
            for (int j=step; j<image.cols-step; j+=step)
            {
                // x,y,radius
//                kps.push_back(KeyPoint(float(j), float(i), float(step)));
//                keypoints.push_back(KeyPoint(int(j), int(i), int(step)))
                keypoints->push_back(KeyPoint(float(j), float(i), float(step)));
            }
        }
        f2d->detectAndCompute(*M, noArray(), *keypoints, *descriptors);
//        cv::detail::ImageFeatures imageFeatures;
        cv::detail::computeImageFeatures(f2d,image, result.imageFeatures);

        for (size_t i = 0; i < (*keypoints).size(); i++) {
            result.keypoints.push_back((*keypoints)[i]);
        }
        (*descriptors).copyTo(result.descriptors);
        (*M).copyTo(result.image);
        (*M).release();
        delete M;
        (*keypoints).clear();
        delete keypoints;
        (*descriptors).release();
        delete descriptors;
        f2d->clear();
        return 1;
    }
    catch (...) { return 0; }
}

//! original homography matrix calculator
int checkimage(imagestatus &result, featuredata& basedata, Mat& image, int direction, double cutsize, double compression_ratio)
{
    Mat *image_ = new Mat();
    image.copyTo(*image_);
    try {
        featuredata *checkdata = new featuredata();
        getfeaturedata(*checkdata, *image_, direction, cutsize, compression_ratio);//获取检测图特征信息

        if ((*checkdata).keypoints.size() < GOODMATCHNUMBER) {
            result.direction_status = -1;
            result.homo = (Mat_<double>(3, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
            checkdata->descriptors.release();
            checkdata->image.release();
            checkdata->keypoints.clear();
            delete checkdata;
            (*image_).release();
            delete image_;
            return 0;
        }

        vector<DMatch> *goodmatchpoints = new vector<DMatch>;
        get_good_match_point(*goodmatchpoints, basedata.descriptors, (*checkdata).descriptors);//筛选匹配点

        homoandmask *hmdata = new homoandmask;
        gethomoandmask(*hmdata, basedata.keypoints, (*checkdata).keypoints, *goodmatchpoints);//计算单应性矩阵

        vector<DMatch> *lastmatchpoints = new vector<DMatch>;
        //        vector<Point2f> *ImagePoints1 = new vector<Point2f>, *ImagePoints2 = new vector<Point2f>;
        //计算最后匹配上的点的数量
        for (size_t i = 0; i < (*hmdata).mask.size(); i++) {
            if ((*hmdata).mask[i] != (uchar)0) {
                (*lastmatchpoints).push_back((*goodmatchpoints)[i]);
                //                (*ImagePoints1).push_back(basedata.keypoints[(*goodmatchpoints)[i].queryIdx].pt);
                //                (*ImagePoints2).push_back((*checkdata).keypoints[(*goodmatchpoints)[i].trainIdx].pt);
            }
        }

        //        cout << "Good Match: " << (*goodmatchpoints).size() << endl;
        //        cout << "Last Match: " << (*lastmatchpoints).size() << endl;;
        //        Mat img_matches;
        //        drawMatches( basedata.image, basedata.keypoints, checkdata->image, checkdata->keypoints, *goodmatchpoints, img_matches, Scalar::all(-1),
        //                     Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        //        imwrite("good_matches.jpg", img_matches );
        //
        //        Mat img_last_matches;
        //        drawMatches( basedata.image, basedata.keypoints, checkdata->image, checkdata->keypoints, *lastmatchpoints, img_last_matches, Scalar::all(-1),
        //                     Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        //        imwrite("last_matches.jpg", img_last_matches );

        //        result.direction_status = checkimagestatus((*checkdata).image, (*hmdata).homo, direction, 1 - cutsize);
        result.direction_status = 0;
        if ((*lastmatchpoints).size() < GOODMATCHNUMBER) {
            result.direction_status = 0;;
        }

        if ((*lastmatchpoints).size() >= GOODMATCHNUMBER) {
            result.direction_status = 1;
        }

        (*hmdata).homo.copyTo(result.homo);
        if (result.direction_status < 1) { result.homo = (Mat_<double>(3, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0); }
        checkdata->descriptors.release();
        checkdata->image.release();
        checkdata->keypoints.clear();
        delete checkdata;
        hmdata->mask.clear();
        hmdata->homo.release();
        delete hmdata;
        (*image_).release();
        delete image_;
        (*goodmatchpoints).clear();
        delete goodmatchpoints;
        return 1;
    }
    catch (exception) {
        result.direction_status = result.direction_status = -1;;
        (*image_).release();
        delete image_;
        return 0;
    }
    catch (Exception) {
        result.direction_status = result.direction_status = -1;;
        (*image_).release();
        delete image_;
        return 0;
    }
}


int computepoint(Point2f &dstpoint, Point2f &oripoint, Mat &homo)
{
    try {
        Mat *point = new Mat();
        *point = (Mat_<double>(3, 1) << oripoint.x, oripoint.y, 1.0);
        Mat *point_ = new Mat();
        *point_ = homo * (*point);
        *point_ /= abs((*point_).at<double>(2, 0)) + tmmin;
        dstpoint.x = (*point_).at<double>(0, 0);
        dstpoint.y = (*point_).at<double>(1, 0);
        (*point).release();
        delete point;
        (*point_).release();
        delete point_;
        return 1;
    }
    catch (exception) { return 0; }
    catch (Exception) { return 0; }
}


int forback(vector<Point2f> &vp, vector<Point2f> &vp_)
{
    boxdata *box = new boxdata;
    get_boxdata(*box, vp);
    double S = (box->xmax - box->xmin) * (box->ymax - box->ymin);
    boxdata *box_ = new boxdata;
    get_boxdata(*box_, vp_);
    double S_ = (box_->xmax - box_->xmin) * (box_->ymax - box_->ymin);
    delete box;
    delete box_;
    if ((S_ - S) > (0.2*S)) {
        //cout << "FORWORD" << endl;
        return 1000;
    }
    if ((S_ - S) < (-0.2*S)) {
        //cout << "BACKWORD" << endl;
        return 2000;
    }
    return 0;
}


int updown(Point2f &p, Point2f &p_, double d)
{
    double dy = p.y - p_.y;
    int result = 0;
    if (dy < -d) {
        result += 100;
    }
    if (dy > d) {
        result += 200;
    }
    return result;
}


int leftright(Point2f &p, Point2f &p_, double d)
{
    double dx = p.x - p_.x;
    int result = 0;
    if (dx < -d) {
        result += 10;
    }
    if (dx > d) {
        result += 20;
    }
    return result;
}


int rotate(Point2f &p1, Point2f &p2, double d)
{
    double dy = p1.y - p2.y;
    //cout << dy << endl;
    int result = 0;
    if (dy < -d) {
        //cout << "CLOCKWISE" << endl;
        result += 10000;
    }
    if (dy > d) {
        //cout << "COUNTERCLOCKWISE" << endl;
        result += 20000;
    }
    return result;
}


int checkimagestatus(Mat& Image, Mat& homo, int direction, double cutsize)
{
    int result = 0;
    try {
        double R = Image.rows;
        double C = Image.cols;
        Point2f  p1(0.0, 0.0), p2(C, 0.0), p3(0.0, R), p4(C, R), p5(C / 2, R / 2), p6(0, R / 2), p7(C, R / 2);
        //Point2f p1_ = computepoint(p1, homo);
        //Point2f p2_ = computepoint(p2, homo);
        //Point2f p3_ = computepoint(p3, homo);
        //Point2f p4_ = computepoint(p4, homo);
        Point2f p5_;
        computepoint(p5_, p5, homo);
        //Point2f p6_ = computepoint(p6, homo);
        //Point2f p7_ = computepoint(p7, homo);
        double PX = 0.01 * C, PY = 0.01 * R;
        if (direction == 0 || direction == 1) {
            PX = 0.06 * C * cutsize;
            PY = 0.04 * R * cutsize;
        }
        else if (direction == 2 || direction == 3) {
            PX = 0.04 * C * cutsize;
            PY = 0.06 * R * cutsize;
        }
        result += leftright(p5, p5_, PX);
        result += updown(p5, p5_, PY);
        if (result == 0) {
            result = 1;
        }
        //result += rotate(p6_, p7_, 2.0);
        //if (result == 1) {
        //    vector<Point2f> vp, vp_;
        //    vp.push_back(p1);
        //    vp.push_back(p2);
        //    vp.push_back(p3);
        //    vp.push_back(p4);
        //    vp_.push_back(p1_);
        //    vp_.push_back(p2_);
        //    vp_.push_back(p3_);
        //    vp_.push_back(p4_);
        //    result = forback(vp, vp_);
        //}
        //if (result == 0) {
        //    result = 1;
        //}
        return result;
    }
    catch (exception) { result = -1; }
    catch (Exception) { result = -1; }
    return result;
}


int cutimage(Mat& result, Mat& image, int xmin, int ymin, int xmax, int ymax)
{
    try {
        Rect rect(xmin, ymin, xmax - xmin, ymax - ymin);
        (image(rect)).copyTo(result);
        return 1;
    }
    catch (...) { return 0; }
}


int LoadImage(Mat& result, Mat& image, int direction, double cutsize, double compression_ratio)
{
    try {
        int c = image.cols;
        int r = image.rows;
        int cols;
        int rows;
        if (direction == 0 || direction == 1)
        {
//            if (r > c)
//            {
//                cutimage(result, image, 0, (int)(r * 0.2), c, (int)(r * 0.8));
//            }
//            else
//            {
//                result = image;
//            }
            result = image;
            double ratio = double(result.rows) / n_max;
            cols = (int)(result.cols / ratio);
            rows = n_max;
        }
        else if (direction == 2 || direction == 3)
        {
//            if (r < c)
//            {
//                cutimage(result, image, (int)(c * 0.2), 0, (int)(c * 0.8), r);
//            }
//            else
//            {
//                result = image;
//            }
            result = image;
            double ratio = double(result.cols) / n_max;
            cols = n_max;
            rows = (int)(result.rows / ratio);
        }
        else
        {
            return 0;
        }

        Size size = Size(cols, rows);
        resize(result, result, size, 0, 0, INTER_AREA);

        switch (direction) {
            case 0:
                cutimage(result, result, 0, 0, (int)(cols * cutsize), rows);
                break;
            case 1:
                cutimage(result, result, (int)(cols * (1 - cutsize)), 0, cols, rows);
                break;
            case 2:
                cutimage(result, result, 0, 0, cols, (int)(rows * cutsize));
                break;
            case 3:
                cutimage(result, result, 0, (int)(rows * (1 - cutsize)), cols, rows);
                break;
            default:
                break;
        }
        return 1;
    }
    catch (...) { return 0; }
}


int get_good_match_point(vector<DMatch> &result, Mat& descriptors1, Mat& descriptors2)
{
    result.clear();
    try {
        BFMatcher *matcher = new BFMatcher;
        //        FlannBasedMatcher *matcher = new FlannBasedMatcher();
        vector<vector<DMatch>> matchePoints12;
        if (descriptors1.rows < 1 || descriptors2.rows < 1) { return 0; }
        (*matcher).knnMatch(descriptors1, descriptors2, matchePoints12, 2);
        //        double mindist = matchePoints12[0][0].distance;
        for (size_t i = 0; i < matchePoints12.size(); i++) {
            //            mindist = MIN(matchePoints12[i][0].distance, mindist);
            if (matchePoints12[i][0].distance < 0.75 * matchePoints12[i][1].distance) {
                result.push_back(matchePoints12[i][0]);
            }
        }
        (*matcher).clear();
        delete matcher;
        matchePoints12.clear();
        return 1;
    }
    catch (...) { return 0; }
}


int gethomoandmask_v2(homoandmask &result, vector<KeyPoint> &keyPts1, vector<KeyPoint> &keyPts2, vector<DMatch> &GoodMatchePoints, int direction, Mat& image, double cutsize, int match_num)
{
    result.mask.clear();
    try {
        vector<Point2f> *imagePoints1 = new vector<Point2f>, *imagePoints2 = new vector<Point2f>;
        int h_ = image.rows;
        int w_ = image.cols;
        double ratio;
        double delta = 0;
        if (direction == 0 or direction == 1) {
            ratio = (double)(h_) / n_max;
            int w = (int)(w_ / ratio);
            delta = w * (1 - cutsize);
        } else {
            ratio = (double)(w_) / n_max;
            int h = (int)(h_ / ratio);
            delta = h * (1 - cutsize);
        }

        if (GoodMatchePoints.size() < match_num) { return 0; }
        for (auto & GoodMatchePoint : GoodMatchePoints) {
            Point2f pt1 = keyPts1[GoodMatchePoint.queryIdx].pt;
            Point2f pt2 = keyPts2[GoodMatchePoint.trainIdx].pt;


            if (direction == 0) {
                pt1.x += delta;
            } else if (direction == 1) {
                pt2.x += delta;
            } else if (direction == 2) {
                pt1.y += delta;
            } else if (direction == 3) {
                pt2.y += delta;
            }

            pt1.x = pt1.x * ratio;
            pt1.y = pt1.y * ratio;
            pt2.x = pt2.x * ratio;
            pt2.y = pt2.y * ratio;


//            cout << "@@@@" << pt1 << " " << pt2 << endl;

            (*imagePoints1).push_back(pt1);
            (*imagePoints2).push_back(pt2);
        }

        if ((*imagePoints1).size() != (*imagePoints2).size()
            && (*imagePoints1).size() < match_num
            && (*imagePoints2).size() < match_num) {
            (*imagePoints1).clear();
            delete imagePoints1;
            (*imagePoints2).clear();
            delete imagePoints2;
            return 0;
        }
        vector<uchar> mask;

        Mat homo = (Mat_<double>(2, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0);
        mask.clear();
        try {
            homo = findHomography(*imagePoints1, *imagePoints2, RANSAC, 5.0, mask);
            if (!homo.empty() && homo.rows == 3 && homo.cols == 3) {
                Rect rect(0, 0, homo.cols, homo.rows);
                homo.copyTo(result.homo(rect));

//                cout << "####" << result.homo << endl;

            }
            else { mask.clear(); }
        }
        catch (...) { mask.clear(); }
        homo.release();

        for (size_t i = 0; i < mask.size(); i++) {
            result.mask.push_back(mask[i]);
        }
        (*imagePoints1).clear();
        delete imagePoints1;
        (*imagePoints2).clear();
        delete imagePoints2;
    }
    catch (...) { return 0; }
    return 1;
}


int gethomoandmask(homoandmask &result, vector<KeyPoint> &keyPts1, vector<KeyPoint> &keyPts2, vector<DMatch> &GoodMatchePoints)
{
    result.mask.clear();
    try {
        vector<Point2f> *imagePoints1 = new vector<Point2f>, *imagePoints2 = new vector<Point2f>;
        if (GoodMatchePoints.size() < GOODMATCHNUMBER) { return 0; }
        for (size_t i = 0; i < GoodMatchePoints.size(); i++) {
            (*imagePoints1).push_back(keyPts1[GoodMatchePoints[i].queryIdx].pt);
            (*imagePoints2).push_back(keyPts2[GoodMatchePoints[i].trainIdx].pt);
        }
        if ((*imagePoints1).size() != (*imagePoints2).size()
            && (*imagePoints1).size() < GOODMATCHNUMBER
            && (*imagePoints2).size() < GOODMATCHNUMBER) {
            (*imagePoints1).clear();
            delete imagePoints1;
            (*imagePoints2).clear();
            delete imagePoints2;
            return 0;
        }
        vector<uchar> mask;

        Mat homo = (Mat_<double>(2, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0);
        mask.clear();
        try {

            // 仿射变换
            //            homo = estimateAffine2D(*imagePoints2, *imagePoints1, mask);
            //            if (!homo.empty() && homo.rows == 2 && homo.cols == 3) {
            //                Rect rect(0, 0, homo.cols, homo.rows);
            //                homo.copyTo(result.homo(rect));
            //            }

            // 透视变换
            homo = findHomography(*imagePoints2, *imagePoints1, RANSAC, 5.0, mask);
            if (!homo.empty() && homo.rows == 3 && homo.cols == 3) {
                Rect rect(0, 0, homo.cols, homo.rows);
                homo.copyTo(result.homo(rect));
            }


            else { mask.clear(); }
        }
        catch (Exception) { mask.clear(); }
        catch (exception) { mask.clear(); }
        homo.release();

        for (size_t i = 0; i < mask.size(); i++) {
            result.mask.push_back(mask[i]);
        }
        (*imagePoints1).clear();
        delete imagePoints1;
        (*imagePoints2).clear();
        delete imagePoints2;
        return 1;
    }
    catch (...) { return 0; }
}


int get_boxdata(boxdata &result, vector<Point2f>& points)
{
    if (points.size() < 1) {
        return 0;
    }
    result.xmin = points[0].x;
    result.ymin = points[0].y;
    result.xmax = points[0].x;
    result.ymax = points[0].y;
    for (size_t i = 0; i < points.size(); i++) {
        result.xmin = MIN(result.xmin, points[i].x);
        result.ymin = MIN(result.ymin, points[i].y);
        result.xmax = MAX(result.xmax, points[i].x);
        result.ymax = MAX(result.ymax, points[i].y);
    }
    return 1;
}


void triangulation(Mat R, Mat t){
    Mat T1 = (Mat_<double>(3,4)<<
                               1,0,0,0,
            0,1,0,0,
            0,0,1,0);

    Mat T2 = (Mat_<double>(3,4)<<
                               R.at<double>(0,0),R.at<double>(0,1),R.at<double>(0,2), t.at<double>(0,0),
            R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1,0),
            R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2,0));

    Mat K = ( Mat_<double> ( 3,3 ) << 339.2815377288591, 0, 168.5,
            0, 339.2815377288591, 300,
            0, 0, 1);




}

stitch_status *previous_status;

int check_image_for_IOS(stitch_status &result, featuredata& basedata, Mat& image, int direction, double cutsize, double compression_ratio, int match_num1, int match_num2, double threshold1, double threshold2, int rows, int cols) {
    stitch_status *current_result = new stitch_status();
    check_image_v2(*current_result,basedata,image,direction,cutsize,compression_ratio,match_num1,match_num2,threshold1,threshold2);

    if (current_result->direction_status != 2){
        if (previous_status->direction_status ==2){
            for (int i =0 ;i<  previous_status->corner.size(); i++ ){
                current_result->corner[i] = previous_status->corner[i];
            }

        }
    }
    previous_status->direction_status = current_result->direction_status;
    result.corner = current_result->corner;
    result.direction_status=current_result->direction_status;

    float x_scale = float(1080.0)/float(cols);
    float y_scale = float(1920.0)/float(rows);
    for (int i =0 ;i<  current_result->corner.size(); i++ ){
        previous_status->corner[i] = current_result->corner[i];


        result.corner[i].x =result.corner[i].x*x_scale;
        result.corner[i].y =result.corner[i].y*y_scale;

    }



}


