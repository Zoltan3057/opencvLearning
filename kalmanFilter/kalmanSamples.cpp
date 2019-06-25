/***********************************************************************
 *
 * Mario Read on   : 2019-06-24 16:16
 * Filename      : kalman.cpp
 * Function      : 一维的卡尔曼滤波
 *
 *************************************************************************/

#include <iostream>
#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <stdio.h>

using namespace cv;

static inline Point calcPoint(Point2f center, double R, double angle)
{
    return center + Point2f((float)cos(angle), (float)-sin(angle))*(float)R;
}

static void help()
{
    printf( "\nExample of c calls to OpenCV's Kalman filter.\n"
"   Tracking of rotating point.\n"
"   Rotation speed is constant.\n"
"   Both state and measurements vectors are 1D (a point angle),\n"
"   Measurement is the real point angle + gaussian noise.\n"
"   The real and the estimated points are connected with yellow line segment,\n"
"   the real and the measured points are connected with red line segment.\n"
"   (if Kalman filter works correctly,\n"
"    the yellow segment should be shorter than the red one).\n"
            "\n"
"   Pressing any key (except ESC) will reset the tracking with a different speed.\n"
"   Pressing ESC will stop the program.\n"
            );
}

int main(int, char**)
{

    help();
    Mat img(500, 500, CV_8UC3);

    // 两个状态量,一个测量量,0个控制量
    KalmanFilter KF(2, 1, 0);

    Mat state(2, 1, CV_32F); /* (phi, delta_phi) */
    Mat processNoise(2, 1, CV_32F);

    Mat measurement = Mat::zeros(1, 1, CV_32F);

    // 读取键盘输入
    char code = (char)-1;

    for(;;)
    {

        randn( state, Scalar::all(0), Scalar::all(0.1) );

        // 状态变化 此为真实值
        // [1,1] * [theta     ] = [theta+deltaTheta]
        // [0,1]   [deltaTheta]   [deltaTheta]
        KF.transitionMatrix = (Mat_<float>(2, 2) << 1, 1, 0, 1);    // 转移矩阵A

        setIdentity(KF.measurementMatrix);// 测量矩阵
        setIdentity(KF.processNoiseCov, Scalar::all(1e-5));// 过程噪声
        setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));// 测量噪声
        setIdentity(KF.errorCovPost, Scalar::all(1));// 最小均方误差

        randn(KF.statePost, Scalar::all(0), Scalar::all(0.1));// 系统初始状态

        for(;;)
        {

            // 计算真实值
            Point2f center(img.cols*0.5f, img.rows*0.5f);
            float R = img.cols/3.f;
            double stateAngle = state.at<float>(0);
            Point statePt = calcPoint(center, R, stateAngle);

            // 预测,第1,2个式子
            Mat prediction = KF.predict();
            double predictAngle = prediction.at<float>(0);
            Point predictPt = calcPoint(center, R, predictAngle);

            // generate measurement, 得到测量值[造的数据]
            randn( measurement, Scalar::all(0), Scalar::all(KF.measurementNoiseCov.at<float>(0)));
            measurement += KF.measurementMatrix*state;
            double measAngle = measurement.at<float>(0);
            Point measPt = calcPoint(center, R, measAngle);

            // plot points
            #define drawCross( center, color, d )                                        \
                line( img, Point( center.x - d, center.y - d ),                          \
                             Point( center.x + d, center.y + d ), color, 1, LINE_AA, 0); \
                line( img, Point( center.x + d, center.y - d ),                          \
                             Point( center.x - d, center.y + d ), color, 1, LINE_AA, 0 )

            img = Scalar::all(0);// 图像赋值
            // 真实值
            drawCross( statePt, Scalar(255,255,255), 6 );
            // 测量值,有噪声(0,R)
            drawCross( measPt, Scalar(0,0,255), 6 );
            // 第一,二步的预测值
            drawCross( predictPt, Scalar(0,255,0), 6 );

            line( img, statePt, measPt, Scalar(0,0,255), 3, LINE_AA, 0 );
            line( img, statePt, predictPt, Scalar(0,255,255), 3, LINE_AA, 0 );

            //returns uniformly distributed integer random number from [a,b) range
            //if(theRNG().uniform(0,4) != 0){

            if(theRNG().uniform(0,100) >= 0){
                std::cout << "correct" << std::endl;
                // 更新,第3,4,5式子
                KF.correct(measurement);
            }else{
                std::cout << "don't correct" << std::endl;
            }

            /**
            theRNG().uniform(0,100);
            KF.correct(measurement);
            */

            randn( processNoise, Scalar(0), Scalar::all(sqrt(KF.processNoiseCov.at<float>(0, 0))));
            state = KF.transitionMatrix*state + processNoise;

            imshow( "Kalman", img );
            code = (char)waitKey(100);

            if( code > 0 )
                break;
        }
        if( code == 27 || code == 'q' || code == 'Q' )
            break;
    }

    return 0;
}
