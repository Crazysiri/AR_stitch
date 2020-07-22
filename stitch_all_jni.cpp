//
// Created by Hao Bai on 2020/5/9.
//

#include "panorama_jni.h"
// #ifndef PANORAMA_PANORAMA_CHECK_H
// #define PANORAMA_PANORAMA_CHECK_H
#include "panorama_check.h"
// #endif //PANORAMA_PANORAMA_CHECK_H
#include "newstitchcheck.h"
#include "newstitchcheck.cpp"
bool isHorizontail_Global = false;

store_each *all_param = new store_each();
featuredata temp;
jint previous[9];
// store_each all_param_result;
extern "C" {
jintArray matToBitmapArray(JNIEnv *env, const cv::Mat &image) {
    jintArray resultImage = env->NewIntArray(image.total()+2);
    jint *_data = new jint[image.total()+2];
    for (int i = 0; i < image.total(); i++) {
        char r = image.data[3 * i + 2];
        char g = image.data[3 * i + 1];
        char b = image.data[3 * i + 0];
        char a = (char) 255;
        _data[i] = (((jint) a << 24) & 0xFF000000) + (((jint) r << 16) & 0x00FF0000) +
                   (((jint) g << 8) & 0x0000FF00) + ((jint) b & 0x000000FF);
    }
    _data[image.total()+1] = image.rows;
    _data[image.total()] = image.cols;
    env->SetIntArrayRegion(resultImage, 0, image.total()+2, _data);
    delete[]_data;

    return resultImage;
}



JNIEXPORT jintArray JNICALL
Java_com_trax_jcall_AlgorithmNativeCarrier_generateResult(JNIEnv *env,
                                                          jobject obj,
                                                          jobject image,
                                                          jint image_num,
                                                          jboolean isHorizontail) {
    try {


        jclass jmat = (env)->FindClass("org/opencv/core/Mat");
        jmethodID getNativeObjAddr = (env)->GetMethodID(jmat, "getNativeObjAddr", "()J");
        jlong getimage = (env)->CallLongMethod(image, getNativeObjAddr, NULL);
        cv::Mat myimage = cv::Mat();
        myimage = *(cv::Mat *) getimage;

        if((int)image_num==1){
            isHorizontail_Global = (bool)isHorizontail;
        }
        clock_t startTime,endTime;
        startTime = clock();
//        jintArray result = matToBitmapArray(env, myimage);
//        return result;
        if (myimage.empty()) {
            jintArray errorArray = env->NewIntArray(1);
            return errorArray;
        }
        store_each temop_all_param(*all_param);
        myimage = resize_the_input(myimage);
        all_param->full_imgs.push_back(myimage);

        // endTime = clock();
        // double process_time = (double)(endTime - startTime) / CLOCKS_PER_SEC;
        // LOGW("########## process_time = %d", process_time);
        startTime = clock();
        if (isHorizontail_Global){

            generate_result_horizontail(*all_param, image_num);
        }else{
            generate_result(*all_param, image_num);
        }
        // endTime = clock();
        // process_time = (double)(endTime - startTime) / CLOCKS_PER_SEC;
        // LOGW("########## process_time = %d", process_time);

        myimage.release();
        if (image_num == 0) {
            jintArray zeroArray = env->NewIntArray(1);
            return zeroArray;
        }
// if status is 1 error re run
        if (all_param->status == 1) {
            *all_param = temop_all_param;
            jintArray errorArray = env->NewIntArray(1);

            return errorArray;
        } else if (all_param->status == 0) {
            startTime = clock();
            jintArray result = matToBitmapArray(env, all_param->result_stitched_img);


            // endTime = clock();
            // process_time = (double)(endTime - startTime) / CLOCKS_PER_SEC;
            // LOGW("########## change to bitmap time = %d", process_time);
            return result;
        }

    } catch (...) {
        jintArray errorArray = env->NewIntArray(1);
        return errorArray;
    }

}


JNIEXPORT jint JNICALL
Java_com_trax_jcall_AlgorithmNativeCarrier_resetStitch(JNIEnv *,
                                                       jobject){
    try {
        reset_it(*all_param);
        isHorizontail_Global = false;
        return 0;
    }catch(...){
        return 1;
    }

}


JNIEXPORT jint JNICALL
Java_com_trax_jcall_AlgorithmNativeCarrier_rollBack(JNIEnv *,
                                                    jobject,
                                                    jint index){
    try {
        int isOK = roll_back_with_index_two_img(*all_param, (int)index);
        if (isOK == 0){
            return 0;
        }else{
            return 1;
        }
    }catch(...){
        return 1;
    }
}

//JNIEXPORT jintArray JNICALL
//Java_com_data100_taskmobile_ui_main_activity_MainActivity_DeleteAndFree(JNIEnv *,
//                                                                        jobject){
//    try {
//        free_it(all_param);
//        return 0;
//    }catch(...){
//        return 1;
//    }
//}
JNIEXPORT jintArray JNICALL
Java_com_trax_jcall_AlgorithmNativeCarrier_generateResultPanoStitch(JNIEnv *env,
                                                                    jobject obj,
                                                                    jobject image,
                                                                    jint image_num,
                                                                    jboolean isHorizontail,
                                                                    jint rows_total){
    try {


        jclass jmat = (env)->FindClass("org/opencv/core/Mat");
        jmethodID getNativeObjAddr = (env)->GetMethodID(jmat, "getNativeObjAddr", "()J");
        jlong getimage = (env)->CallLongMethod(image, getNativeObjAddr, NULL);
        cv::Mat myimage = cv::Mat();
        myimage = *(cv::Mat *) getimage;

        if((int)image_num==1){
            isHorizontail_Global = (bool)isHorizontail;
        }


//        jintArray result = matToBitmapArray(env, myimage);
//        return result;
        if (myimage.empty()) {
            jintArray errorArray = env->NewIntArray(1);
            return errorArray;
        }
        store_each temop_all_param(*all_param);
        myimage = resize_the_input(myimage);
        all_param->full_imgs.push_back(myimage);

        int return_thing;
        if (isHorizontail_Global){

            return_thing = must_generate_one(*all_param, image_num,(bool)isHorizontail_Global,(int)rows_total );
        }else{
            return_thing = must_generate_one(*all_param, image_num, (bool)isHorizontail_Global,(int)rows_total);
        }

        myimage.release();
        if (image_num == 0) {
            jintArray zeroArray = env->NewIntArray(1);
            return zeroArray;
        }
// if status is 1 error re run
        if (return_thing == 1) {
            *all_param = temop_all_param;
            jintArray errorArray = env->NewIntArray(1);

            return errorArray;
        } else if (return_thing == 0) {
            jintArray result = matToBitmapArray(env, all_param->result_stitched_img);

            return result;
        }

    } catch (...) {
        jintArray errorArray = env->NewIntArray(1);
        return errorArray;
    }


}


JNIEXPORT jboolean JNICALL
Java_com_trax_jcall_AlgorithmNativeCarrier_getfeaturedata(JNIEnv *env,
                                                          jobject obj,
                                                          jobject image,
                                                          jint direction,
                                                          jdouble cutsize,
                                                          jdouble compression_ratio)
{
    try
    {
        jclass jmat = (env)->FindClass("org/opencv/core/Mat");
        jmethodID getNativeObjAddr = (env)->GetMethodID(jmat, "getNativeObjAddr", "()J");
        jlong getimage = (env)->CallLongMethod(image, getNativeObjAddr, NULL);
        cv::Mat myimage = Mat();
        myimage = *(Mat *)getimage;
        //LOGE("print %d %d", myimage.rows,myimage.cols);
        //LOGE("print %d", myimage.channels());

        if (myimage.empty())
        {
            jint a = 0;
            return (jboolean)a;
        }

        temp.keypoints.clear();
        temp.descriptors.release();
        temp.image.release();
        temp.full_image.release();


        getfeaturedata(temp, myimage, (int)direction, (double)cutsize, (double)compression_ratio);
//        LOGE("print1: %d %d", temp.image.rows,temp.image.cols);
//        LOGE("print2: %d -- %lf -- %lf", direction,cutsize,compression_ratio);
        if (temp.imageFeatures.descriptors.empty())
        {
            jint a = 0;
            return (jboolean)a;
        }
        else
        {
            jint a = 1;
            return (jboolean)a;
        }

    }

    catch(exception)
    {
        jint a = 0;
        return (jboolean)a;
    }
}

JNIEXPORT jintArray JNICALL
Java_com_trax_jcall_AlgorithmNativeCarrier_checkimage(JNIEnv *env,
                                                      jobject obj,
                                                      jobject image,
                                                      jint direction,
                                                      jdouble cutsize,
                                                      jdouble compression_ratio,
                                                      jint min_feature_num,
                                                      jint perfect_feature_num,
                                                      jfloat warp_image_size_scale_threshold,
                                                      jfloat direction_threshold)
{
    cv::Mat *myimage = new cv::Mat;
    try
    {
        jclass jmat = (env)->FindClass("org/opencv/core/Mat");
        jmethodID getNativeObjAddr = (env)->GetMethodID(jmat, "getNativeObjAddr", "()J");
        jlong getimage = (env)->CallLongMethod(image, getNativeObjAddr, NULL);
        (*(Mat *)getimage).copyTo(*myimage);
        if ((*myimage).empty())
        {
            jintArray kk = env -> NewIntArray(9);
            jint p[9] = {-1, 0, 0, 0, 0, 0, 0, 0, 0};
            env->SetIntArrayRegion(kk, 0, 9, p);
            //delete []p;
            return kk;
        }
        stitch_status *result = new stitch_status;
        cout<<temp.imageFeatures.keypoints.size()<<endl;
        check_image_v2(*result, temp, *myimage, (int)direction, (double)cutsize, (double)compression_ratio, (int)min_feature_num, (int)perfect_feature_num, (float)warp_image_size_scale_threshold, (float)direction_threshold);
        (*myimage).release();
        delete myimage;
        jintArray kk = env -> NewIntArray(9);

        jint p[9];
        int count = 0;
        p[count++] = (jint)(*result).direction_status;
        for (size_t i = 0; i < result->corner.size(); i++) {
            Point2f pt = result->corner[i];
            p[count++] = (jint)pt.x;
            p[count++] = (jint)pt.y;
        }
        if(p[0] != 2){
            if(previous[0] == 2){
                for(int i = 0;i < 9;i++ ){
                    p[i] = previous[i];
                }

                previous[0] = 0;
            }else{
                for(int i = 0;i < 9;i++ ){
                    previous[i] = p[i];
                }
            }
        }else{
            for(int i = 0;i < 9;i++ ){
                previous[i] = p[i];
            }
        }
        env->SetIntArrayRegion(kk, 0, 9, p);


        delete result;

        //delete []p;
        return kk;
    }

    catch(...)
    {
        // LOGE("^&**&^^&*  print checkimage1 ERROR");
        delete myimage;
        jintArray kk = env -> NewIntArray(9);
        jint p[9] = {-1, 0, 0, 0, 0, 0, 0, 0, 0};
        env->SetIntArrayRegion(kk, 0, 9, p);
        //delete []p;
        return kk;
    }
}



}
