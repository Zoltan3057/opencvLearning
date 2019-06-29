#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

#include <iostream>

using namespace cv;
using namespace cv::ml;
using namespace std;

int main(){

	float trainingData[8][3] = { {6, 180, 12}, {5.92, 190, 11}, {5.58, 170, 12}, {5.92, 165, 10},
								{5, 100, 6}, {5.5, 150, 8},{5.42, 130, 7}, {5.75, 150, 9}};
	Mat trainingDataMat(8, 3, CV_32FC1, trainingData);
    //cout << trainingDataMat << endl;

    int responses[8] = {'M','M','M','M','F','F','F','F'};
    Mat responsesMat(8,1,CV_32SC1,responses);
    //cout << responsesMat << endl;

#if 1

    Ptr<NormalBayesClassifier> nbc = NormalBayesClassifier::create();
    Ptr<TrainData> trainData = TrainData::create(trainingDataMat, ROW_SAMPLE, responsesMat);

    /**
     * TrainData 类方法:
    Mat debugTrain = trainData->getTrainSamples();
    cout << debugTrain << endl;
    Mat debugResponse = trainData->getTrainResponses();
    cout << debugResponse << endl;
    Mat debugLabels = trainData->getClassLabels();
    cout << debugLabels << endl;
    Mat debugIdx = trainData->getVarIdx();
    cout << debugIdx << endl;
    int nclasses = (int)debugLabels.total();
    cout << "nclasses: " << nclasses << endl;
    */

    nbc->train(trainData);

#else
    nbc->train(trainingDataMat, ROW_SAMPLE, responsesMat);

#endif

    float myData[3] = {6,130,8};
    Mat myDataMat(1,3,CV_32FC1,myData);
    float r = nbc->predict(myDataMat);

    cout << "result: " << r <<", "  << (char)r<<endl;

    return 0;
}
