#include "newstitchcheck.h"
#include "exception"
#include <stdlib.h>
#define GOODMATCHNUMBER 20
#define n_max 600;
double work_megapix = 0.6;
double seam_megapix = 0.1;
double compose_megapix = -1;
float conf_thresh = 1.f;
float match_conf = 0.3f;
float blend_strength = 5;
double work_scale = 1, seam_scale = 1, compose_scale = 1;
bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;
float seam_work_aspect = 1.0f;
Ptr<Feature2D> finder = xfeatures2d::SURF::create(1000);
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

cv::Point2f calcWarpedPoint(
        const cv::Point2f& pt,
        InputArray K1,                // Camera K parameter
        InputArray R1,                // Camera R parameter
        InputArray K2,                // Camera K parameter
        InputArray R2,                // Camera R parameter
        Ptr<cv::detail::RotationWarper> warper,  // The Rotation Warper
        const std::vector<cv::Point> &corners,
        const std::vector<cv::Size> &sizes)
{
    cv::Point2f  dst = warper->warpPoint(pt, K1, R1);
    cv::Point2f  tl = cv::detail::resultRoi(corners, sizes).tl();
    dst = cv::Point2f(dst.x - tl.x, dst.y - tl.y);
    Mat R2_t;
    cv::invert(R2 ,R2_t);
    cv::Point2f  dst2 = warper->warpPoint(dst, K2, R2_t);
    cv::Point2f  t2 = cv::detail::resultRoi(corners, sizes).tl();
    cv::Point2f final_pt = cv::Point2f(dst2.x - t2.x, dst2.y - t2.y);

//    cv::Point2f final_pt = dst2;
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

int check_image_v2(stitch_status &result, featuredata& basedata, Mat& image, int direction, double cutsize, double compression_ratio, int match_num1, int match_num2, double threshold1, double threshold2)
{
    try{
    result.direction_status = 0;
//    work_scale = float(image.rows)/float(800);
    Mat full_v2_image = image.clone();
    cv::Size full_img_size;
    full_img_size.height = image.rows;
    full_img_size.width = image.cols;

    vector<Size> full_img_sizes(2);
    full_img_sizes[0] = full_img_size;
    full_img_sizes[1] = full_img_size;

    work_scale = min(1.0, sqrt(work_megapix * 1e6 / image.size().area()));
    work_scale = float(800)/float(image.rows);
    resize(image, image, Size(), work_scale, work_scale, 5);

    cv::detail::ImageFeatures image2Feature;
    seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / image.size().area()));

    seam_work_aspect = seam_scale / work_scale;
    cv::detail::computeImageFeatures(finder, image, image2Feature);
    resize(image, image, Size(), seam_scale, seam_scale, 5);
    vector<cv::detail::ImageFeatures> features;
    features.push_back(basedata.imageFeatures);
    features.push_back(image2Feature);
    // (3) 创建图像特征匹配器，计算匹配信息

    vector<cv::detail::MatchesInfo> pairwise_matches;
    Ptr<cv::detail::FeaturesMatcher>  matcher = makePtr<cv::detail::BestOf2NearestMatcher>(false, match_conf);
    (*matcher)(features, pairwise_matches);
    matcher->collectGarbage();

    vector<Mat> images(2);
    resize(basedata.image, images[0], Size(), seam_scale, seam_scale, 5);
    resize(image, images[1], Size(), seam_scale, seam_scale, 5);
    //! (4) 剔除外点，保留最确信的大成分
    // Leave only images we are sure are from the same panorama
    vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
    vector<Mat> img_subset;
//    vector<String> img_names_subset;
    vector<Size> full_img_sizes_subset;
    for (size_t i = 0; i < indices.size(); ++i)
    {
//        img_names_subset.push_back(img_names[indices[i]]);
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
        std::cout << "Need more images\n";
        return -1;
    }

    //!(5) 估计 homography
    Ptr<cv::detail::Estimator> estimator = makePtr<cv::detail::HomographyBasedEstimator>();
    vector<cv::detail::CameraParams> cameras;
    if (!(*estimator)(features, pairwise_matches, cameras))
    {
        cout << "Homography estimation failed.\n";
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

    vector<Point> corners(num_images);
//    vector<UMat> masks_warped(num_images);
    vector<UMat> images_warped(num_images);
    vector<Size> sizes(num_images);

    // Warp images and their masks
    Ptr<WarperCreator> warper_creator = makePtr<cv::CylindricalWarper>();
    if (!warper_creator)
    {
        cout << "Can't create the warper \n";
        return 1;
    }

    //! Create RotationWarper
    Ptr<cv::detail::RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));

    //! Calculate warped corners/sizes/mask
    for (int i = 0; i < num_images; ++i)
    {
        Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);
        float swa = (float)seam_work_aspect;
        K(0, 0) *= swa; K(0, 2) *= swa;
        K(1, 1) *= swa; K(1, 2) *= swa;
        corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
        sizes[i] = images_warped[i].size();
//        warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
    }


    images.clear();
    images_warped.clear();


    Mat img_warped, img_warped_s;
    Mat dilated_mask, seam_mask, mask, mask_warped;
//    Ptr<Blender> blender;
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

                Size sz = full_img_sizes[i];
                if (std::abs(compose_scale - 1) > 1e-1)
                {
                    sz.width = cvRound(full_img_sizes[i].width * compose_scale);
                    sz.height = cvRound(full_img_sizes[i].height * compose_scale);
                }

                Mat K;
                cameras[i].K().convertTo(K, CV_32F);
                Rect roi = warper->warpRoi(sz, K, cameras[i].R);

                corners[i] = roi.tl();
                sizes[i] = roi.size();
            }
        }

//        if (abs(compose_scale - 1) > 1e-1)
//            resize(full_img, img, Size(), compose_scale, compose_scale, 5);
//        else
//            img = full_img;
        full_img.release();
//        Size img_size = img.size();

//        Mat K, R;
//        cameras[img_idx].K().convertTo(K, CV_32F);
//        R = cameras[img_idx].R;
//
//        // Warp the current image : img => img_warped
//        warper->warp(img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);
//
//        // Warp the current image mask
//        mask.create(img_size, CV_8U);
//        mask.setTo(Scalar::all(255));
//        warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);
//
//        compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);
        img_warped.convertTo(img_warped_s, CV_16S);
        img_warped.release();

        mask.release();

//        dilate(masks_warped[img_idx], dilated_mask, Mat());
//        resize(dilated_mask, seam_mask, mask_warped.size(), 0, 0, 5);
//        mask_warped = seam_mask & mask_warped;

//        if (!blender)
//        {
//            blender = Blender::createDefault(Blender::MULTI_BAND, false);
//            Size dst_sz = resultRoi(corners, sizes).size();
//            float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
//            if (blend_width < 1.f){
//                blender = Blender::createDefault(Blender::NO, false);
//            }
//            else
//            {
//                MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());
//                mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
//            }
//            blender->prepare(corners, sizes);
//        }
//
//        blender->feed(img_warped_s, mask_warped, corners[img_idx]);
    }



    Mat K0;
    cameras[0].K().convertTo(K0, CV_32F);
    Mat R0;
    cameras[0].R.convertTo(R0, CV_32F);


    Mat K1;
    cameras[1].K().convertTo(K1, CV_32F);
    Mat R1;
    cameras[1].R.convertTo(R1, CV_32F);


//    cv::Point2f my_cpt = cv::Point2f(img_sizes[0].first, img_sizes[0].second);
//    cv::Point my_pt = calcWarpedPoint(my_cpt, K0, R0, K1, R1, warper, corners, sizes);
    vector<pair<int, int> > img_sizes;
    for (int idx = 0; idx < 2; ++idx) {

        img_sizes.push_back(make_pair(basedata.full_image.cols, basedata.full_image.rows));
    }

    cv::Point2f p0 = cv::Point2f(0,0);
    cv::Point2f p1 = cv::Point2f(img_sizes[0].first,0);
    cv::Point2f p2 = cv::Point2f(img_sizes[0].first,img_sizes[0].second);
    cv::Point2f p3 = cv::Point2f(0,img_sizes[0].second);



    cv::Point2f  dst_p0 = warper->warpPoint(p0, K0, R0);
    cv::Point2f  dst_p1 = warper->warpPoint(p1, K0, R0);
    cv::Point2f  dst_p2 = warper->warpPoint(p2, K0, R0);
    cv::Point2f  dst_p3 = warper->warpPoint(p3, K0, R0);

    cv::Point2f  tl = cv::detail::resultRoi(corners, sizes).tl();
    dst_p0 = cv::Point2f(dst_p0.x - tl.x, dst_p0.y - tl.y);
    dst_p1 = cv::Point2f(dst_p1.x - tl.x, dst_p1.y - tl.y);
    dst_p2 = cv::Point2f(dst_p2.x - tl.x, dst_p2.y - tl.y);
    dst_p3 = cv::Point2f(dst_p3.x - tl.x, dst_p3.y - tl.y);
    Mat R1_t;
    cv::invert(R1 ,R1_t);
//    cv::invert(R1_t ,R1_t);
    cv::Point2f  dst2_p0 = warper->warpPoint(dst_p0, K1, R1_t);
    cv::Point2f  dst2_p1 = warper->warpPoint(dst_p1, K1, R1_t);
    cv::Point2f  dst2_p2 = warper->warpPoint(dst_p2, K1, R1_t);
    cv::Point2f  dst2_p3 = warper->warpPoint(dst_p3, K1, R1_t);
//    cv::Point2f  t2 = cv::detail::resultRoi(corners, sizes).tl();
    cv::Point2f final_pt = cv::Point2f(dst2_p0.x - tl.x, dst2_p0.y - tl.y);

//    cv::Point p0_ = calcWarpedPoint(p0, K0, R0, K1, R1, warper, corners, sizes);
//    cv::Point p1_ = calcWarpedPoint(p1, K0, R0, K1, R1, warper, corners, sizes);
//    cv::Point p2_ = calcWarpedPoint(p2, K0, R0, K1, R1, warper, corners, sizes);
//    cv::Point p3_ = calcWarpedPoint(p3, K0, R0, K1, R1, warper, corners, sizes);

    cv::Point p0_ = cv::Point2f(dst2_p0.x - tl.x, dst2_p0.y - tl.y);
    cv::Point p1_ = cv::Point2f(dst2_p1.x - tl.x, dst2_p1.y - tl.y);
    cv::Point p2_ = cv::Point2f(dst2_p2.x - tl.x, dst2_p2.y - tl.y);
    cv::Point p3_ = cv::Point2f(dst2_p3.x - tl.x, dst2_p3.y - tl.y);




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


    polylines(full_v2_image,  ppt, npt, 1, 1, Scalar(0,255,0),2,8,0);



    std::cout << "***************************************" << std::endl;

    std::cout << "\nCheck `result.png`, `result_mask.png` and `result2.png`!\n";
//    imwrite("/home/baihao/jpg/resultxubo.jpg", full_v2_image);

    result.corner = vector<Point2f>({p0_, p1_,
                                     p2_, p3_});

    result.direction_status = 2;

    result.homo = refine_mask;
    return 0;
    } catch (...) {
        result.direction_status = -1;
        return 1;
    }

    //! ending here
    //! ending here
    //! ending here
    //! ending here
    //! ending here
    //! ending here
    //! ending here
    //! ending here
    //! ending here
    //! ending here
    //! ending here
    //! ending here
    //! ending here
    //! ending here



    return 1;
    //! need remove
    featuredata checkdata;
//        FlannBasedMatcher matcher;
    vector<vector<DMatch>> matchePoints12;
    vector<DMatch> goodmatchpoints;
    if (basedata.descriptors.rows < 1 || checkdata.descriptors.rows < 1) {
        return 0;
    }
//        matcher.knnMatch(basedata.descriptors, checkdata.descriptors, matchePoints12, 2);
    for (size_t i = 0; i < matchePoints12.size(); i++) {
        if (matchePoints12[i][0].distance < 0.75 * matchePoints12[i][1].distance) {
            goodmatchpoints.push_back(matchePoints12[i][0]);
        }
    }

    homoandmask hmdata;
    gethomoandmask_v3(hmdata, basedata.keypoints, checkdata.keypoints, goodmatchpoints, direction, image.rows,
                      image.cols, cutsize, match_num1);//计算单应性矩阵
//        cv::detail::AffineWarper::getRTfromHomogeneous()


    vector<DMatch> lastmatchpoints;
    for (size_t i = 0; i < hmdata.mask.size(); i++) {
        if (hmdata.mask[i] != (uchar) 0) {
            lastmatchpoints.push_back(goodmatchpoints[i]);
        }
    }
    float good_point_percentage = (float)lastmatchpoints.size() / (float)hmdata.mask.size();
    cout <<" lastmatchpoints.size()" << lastmatchpoints.size()  <<" points  \n";
    cout <<"hmdata.mask.size()" << hmdata.mask.size()  <<" points  \n";
    cout <<" there are " << good_point_percentage <<" points  \n";
    if (lastmatchpoints.size() < match_num1 ||good_point_percentage  < 0.5) {
        return 0;
    }

    Corner c;
    calculatecorners(c, basedata.image, hmdata.homo);
    vector<Point2f> pointss ;

    pointss.push_back(Point(0,0));
    pointss.push_back(Point(0,1920));
    pointss.push_back(Point(1080,1920));
    pointss.push_back(Point(1080,0));

    Mat output;

//        perspectiveTransform(Mat(pointss), output, invert_camera0_R * ori_camera1_R);

//        cout<< "the points is : "<< output <<"  . ";

    result.homo = hmdata.homo;

    result.corner = vector<Point2f>({Point2f(int(output.at<float>(0,0)), int(output.at<float>(0,1))), Point2f(int(output.at<float>(1,0)), int(output.at<float>(1,1))),
                                     Point2f(int(output.at<float>(2,0)), int(output.at<float>(2,1))), Point2f(int(output.at<float>(3,0)), int(output.at<float>(3,1)))});

//        result.corner = vector<Point2f>({Point2f(c.ltop.x, c.ltop.y), Point2f(c.lbottom.x, c.lbottom.y),
//                                         Point2f(c.rbottom.x, c.rbottom.y), Point2f(c.rtop.x, c.rtop.y)});

    if (lastmatchpoints.size() >= match_num1 && lastmatchpoints.size() < match_num2) {
        result.direction_status = 1;
        return 0;
    }

    if (lastmatchpoints.size() >= match_num2) {
        result.direction_status = 2;

        if (abs(result.corner[0].x - result.corner[2].x) > threshold1 * image.cols ||
            abs(result.corner[0].y - result.corner[2].y) > threshold1 * image.rows ||
            abs(result.corner[1].x - result.corner[3].x) > threshold1 * image.cols ||
            abs(result.corner[1].y - result.corner[3].y) > threshold1* image.rows) {
            result.direction_status = 1;
        }


        switch (direction) {
            case 0:
                if (abs(result.corner[2].y - float(image.rows)) + abs(result.corner[3].y) > threshold2 * image.rows ||
                    abs(result.corner[3].x) > float(image.cols) ||
                    abs(result.corner[2].x) > float(image.cols)) {
                    result.direction_status = -2;
                }
                break;
            case 1:
                if (abs(result.corner[1].y - float(image.rows)) + abs(result.corner[0].y) > threshold2 * image.rows ||
                    result.corner[0].x < 0 ||
                    result.corner[1].x < 0) {
                    result.direction_status = -2;
                }
                break;
            case 2:
                if (abs(result.corner[1].x) + abs(result.corner[2].x - float(image.cols)) > threshold2 * image.cols ||
                    abs(result.corner[1].y) > float(image.rows) ||
                    abs(result.corner[2].y) > float(image.rows)) {
                    result.direction_status = -2;
                }
                break;
            case 3:
                if (abs(result.corner[0].x) + abs(result.corner[3].x - float(image.cols)) > threshold2 * image.cols ||
                    result.corner[0].y < 0 ||
                    result.corner[3].y < 0) {
                    result.direction_status = -2;
                }
                break;
            default:
                break;
        }

        return 0;
    }


}

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
//        Size target_size;
//        target_size.height = 1920;
//        target_size.width = 1080;
//        work_scale = float(800)/float(image.rows);
        work_scale = min(1.0, sqrt(work_megapix * 1e6 / image.size().area()));
        work_scale = float(800)/float(image.rows);
        result.full_image = image;
        resize(image, result.image, Size(), work_scale, work_scale, 5);


        seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / image.size().area()));
        seam_work_aspect = seam_scale / work_scale;
        cv::detail::computeImageFeatures(finder, result.image, result.imageFeatures);
        resize(result.image, result.image, Size(), seam_scale, seam_scale, 5);
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
        Ptr<Feature2D> f2d = xfeatures2d::SURF::create(1000);
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