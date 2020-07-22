#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/affine.hpp>
#include <iostream>
#include "newstitchcheck.h"
#include<time.h>
//#include <Eigen/Core>

using namespace cv;
using namespace std;

int modify();
int main()
{
//    modify();
//    return 1;
    Mat im0 = imread("/home/baihao/jpg/1111111/tracking/jpg/PPMoney_20200720_112833_3780_32_28.jpg", cv::COLOR_BGR2GRAY);
    Mat im1 = imread("/home/baihao/jpg/1111111/tracking/jpg/mat_1595215719182_matchInfo_281.jpg", cv::COLOR_BGR2GRAY);
//    Eigen::Matrix<>
    if(! im0.data) {
        cout << "read image error" << endl;
    }
    featuredata *basedata = new featuredata();

    getfeaturedata(*basedata, im0, 1, 1, 0.5);

    clock_t start, finish;
    start = clock();

    stitch_status *result = new stitch_status();

    check_image_v2(*result, *basedata, im1, 0, 1, 0.5, 10, 20, 3, 1);


    cout << result->direction_status << endl;

    for (size_t i = 0; i < result->corner.size(); i++) {
        Point2f pt = result->corner[i];
        cout << (int)pt.x << ", " << (int)pt.y << ", ";
    }
    cout << endl;

    finish = clock();
    cout << "time cost: " << (double)(finish - start) / CLOCKS_PER_SEC << endl;


    return 0;
}



int modify(){
    VideoCapture capture("/home/baihao/jpg/1111111/tracking/video5.mp4");
//    Mat frame;
//    frame= capture.open("/home/baihao/jpg/1111111/tracking/video.mp4");
//    purecompute();
    if ( !capture.isOpened( ) )
        cout << "fail toopen!" << endl;
    //获取整个帧数
    long totalFrameNumber = capture.get( cv::CAP_PROP_FRAME_COUNT );
    cout << "整个视频共" << totalFrameNumber << "帧" << endl;

    //设置开始帧()
    long frameToStart = 1;
    capture.set( cv::CAP_PROP_POS_FRAMES, frameToStart );
    cout << "从第" << frameToStart << "帧开始读" << endl;
    //设置结束帧
    int frameToStop = 700;

    if ( frameToStop < frameToStart )
    {
        cout << "结束帧小于开始帧，程序错误，即将退出！" << endl;
        return -1;
    }
    else
    {
        cout << "结束帧为：第" << frameToStop << "帧" << endl;
    }
    //获取帧率
    double rate = capture.get( cv::CAP_PROP_FPS );
    cout << "帧率为:" << rate << endl;
    double delay = 1000 / rate;

    bool stop = false;
    //利用while循环读取帧
    //currentFrame是在循环体中控制读取到指定的帧后循环结束的变量
    long currentFrame = frameToStart;

    Mat frame;
    Mat pre_frame;
    while ( !stop )
    {
        //读取下一帧
        if ( !capture.read( frame ) )
        {
            cout << "读取视频失败" << endl;
            return -1;
        }
        if (currentFrame == 2){
            pre_frame = frame.clone();
            featuredata *basedata = new featuredata();

            getfeaturedata(*basedata, pre_frame, 1, 1, 0.5);
            stringstream str;
            str << "/home/baihao/jpg/base_frame/" << currentFrame << ".png";
            imwrite( str.str( ), frame );
        }
        //此处为跳帧操作
        if ( currentFrame % 10 == 0 ) //此处为帧数间隔，修改这里就可以了
        {
            cout << "正在写第" << currentFrame << "帧" << endl;
            stringstream str;
            str << "/home/baihao/jpg/1111111/tracking_frame/" << currentFrame << ".png";        /*图片存储位置*/

            cout << str.str( ) << endl;
//            imwrite( str.str( ), frame );
            featuredata *basedata = new featuredata();
            getfeaturedata(*basedata, pre_frame, 1, 1, 0.5);
            stitch_status *result = new stitch_status();
            check_image_v2(*result, *basedata, frame, 0, 1, 0.5, 10, 20, 3, 1);

            if(result->corner.empty()){
                cout<<"xxxxxxxxxxxxxxxxx\n";
                cout<<"xxxxxxxxxxxxxxxxx\n";
                cout<<"xxxxxxxxxxxxxxxxx\n";
                cout<<"xxxxxxxxxxxxxxxxx\n";
                cout<<"xxxxxxxxxxxxxxxxx\n";
                cout<<"xxxxxxxxxxxxxxxxx\n";
                stringstream str;
                str << "/home/baihao/jpg/notGetFrame/" << currentFrame << ".png";
                imwrite( str.str( ), frame );
            }



//            pre_frame = frame.clone();
        }

        //waitKey(intdelay=0)当delay≤ 0时会永远等待；当delay>0时会等待delay毫秒
        //当时间结束前没有按键按下时，返回值为-1；否则返回按键
        int c = waitKey( delay );
        //按下ESC或者到达指定的结束帧后退出读取视频
        if ( ( char )c == 27 || currentFrame > frameToStop )
        {
            stop = true;
        }
        //按下按键后会停留在当前帧，等待下一次按键
        if ( c >= 0 )
        {
            waitKey( 0 );
        }
        currentFrame++;

    }

}