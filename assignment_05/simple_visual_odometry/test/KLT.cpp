////
//// Created by sicong on 08/11/18.
////
//
//#include <iostream>
//#include <fstream>
//#include <list>
//#include <vector>
//#include <chrono>
//using namespace std;
//
//
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/video/tracking.hpp>
//
//using namespace cv;
//int main( int argc, char** argv )
//{
//
//    if ( argc != 3 )
//    {
//        cout<<"usage: feature_extraction img1 img2"<<endl;
//        return 1;
//    }
//    //-- Read two images
//    Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );
//    Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );
//
//    list< cv::Point2f > keypoints;
//    vector<cv::KeyPoint> kps;
//
//    std::string detectorType = "Feature2D.BRISK";
//    Ptr<FeatureDetector>detector = Algorithm::create<FeatureDetector>(detectorType);
//	detector->set("thres", 100);
//
//
//    detector->detect( img_1, kps );
//    for ( auto kp:kps )
//        keypoints.push_back( kp.pt );
//
//    vector<cv::Point2f> next_keypoints;
//    vector<cv::Point2f> prev_keypoints;
//    for ( auto kp:keypoints )
//        prev_keypoints.push_back(kp);
//    vector<unsigned char> status;
//    vector<float> error;
//    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
//    cv::calcOpticalFlowPyrLK( img_1, img_2, prev_keypoints, next_keypoints, status, error );
//    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
//    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
//    cout<<"LK Flow use time："<<time_used.count()<<" seconds."<<endl;
//
//    // visualize all  keypoints
//    hconcat(img_1,img_2,img_1);
//    for ( int i=0; i< prev_keypoints.size() ;i++)
//    {
//        cout<<(int)status[i]<<endl;
//        if(status[i] == 1)
//        {
//            Point pt;
//            pt.x =  next_keypoints[i].x + img_2.size[1];
//            pt.y =  next_keypoints[i].y;
//
//            line(img_1, prev_keypoints[i], pt, cv::Scalar(0,255,255));
//        }
//    }
//
//    cv::imshow("klt tracker", img_1);
//    cv::waitKey(0);
//
//    return 0;
//}


//
// Created by sicong on 08/11/18.
//

#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <ctime>
#include <chrono>
#include <unordered_set>
#include <math.h>
using namespace std;


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>

using namespace cv;

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

cv::Matx33d Findfundamental(vector<cv::Point2f> prev_subset,vector<cv::Point2f> next_subset){
    Matx33d Norm;
    // cv::Size s = img.size();
    int rows = 500;
    int cols = 700;
    Norm(0,0) = 3.0 / cols;
    Norm(0,1) = 0.0;
    Norm(0,2) = -1.1;
    Norm(1,0) = 0.0;
    Norm(1,1) = 3.0 / rows;
    Norm(1,2) = -1.1;
    Norm(2,0) = 0.0;
    Norm(2,1) = 0.0;
    Norm(2,2) = 1.0;


    for (size_t i = 0; i < prev_subset.size(); i++){
        prev_subset[i].x = prev_subset[i].x * Norm(0,0) - 1.1;
        prev_subset[i].y = prev_subset[i].y * Norm(1,1) - 1.1;
        next_subset[i].x = next_subset[i].x * Norm(0,0) - 1.1;
        next_subset[i].y = next_subset[i].y * Norm(1,1) - 1.1;
    }

    int num = prev_subset.size();
    cv::Mat A(num,9,CV_64FC1, Scalar(1));
    for (int i = 0; i < num; i++){   
        A.at<double>(i,0) = prev_subset[i].x * next_subset[i].x;
        A.at<double>(i,1) = prev_subset[i].y * next_subset[i].x;
        A.at<double>(i,2) = next_subset[i].x;
        A.at<double>(i,3) = prev_subset[i].x * next_subset[i].y;
        A.at<double>(i,4) = prev_subset[i].y * next_subset[i].y;
        A.at<double>(i,5) = next_subset[i].y;
        A.at<double>(i,6) = prev_subset[i].x;
        A.at<double>(i,7) = prev_subset[i].y;
        A.at<double>(i,8) = 1.0;   
    }
    // cout<<"A is"<< A <<endl;
    cv::SVD svd(A);
    cv::Mat VT(svd.vt);
    cv::Mat Fund(VT.row(VT.rows-1));
    
    cv::Matx33d fund;
    for(int i = 0;i < 3; i++){
        for (int j = 0; j < 3; j++){
            fund(i,j) = Fund.at<double>(3*i+j);
        }
    }

    cv::SVD svd1(fund);
    cv::Mat U1(svd1.u);
    cv::Mat W1(svd1.w);
    cv::Mat VT1(svd1.vt);
    W1.at<double>(2) = 0.0;
    cv::Mat w1(3,3,CV_64FC1, Scalar(0));
    for (int i = 0; i < 3; i++){
        w1.at<double>(i,i) = W1.at<double>(i);
    }

    cv::Mat F = (cv::Mat)(U1 * w1 * VT1);
    F = (cv::Mat)((cv::Mat)Norm.t() * F * (cv::Mat)Norm);
    F = F / F.at<double>(2,2); 
    return (Matx33d)F;
}

bool checkinlier(cv::Point2f prev_keypoint,cv::Point2f next_keypoint,cv::Matx33d Fcandidate,double d){
    cv::Matx13d Prev;
    Prev(0) = prev_keypoint.x;
    Prev(1) = prev_keypoint.y;
    Prev(2) = 1.0;
    cv::Matx31d Next;
    Next(0) = next_keypoint.x;
    Next(1) = next_keypoint.y;
    Next(2) = 1.0;

    double tmp = fabs((Prev * Fcandidate * Next)(0));
    if (tmp <= d){
        return true;
    }
    else{
        return false;
    }    
}

cv::Matx33d Findessential(cv::Matx33d F){
    cv::Matx33d K;
    K(0,0) = -517.3;
    K(1,1) = -516.5;
    K(2,2) = 1.0;
    K(0,2) = 318.643040;
    K(1,2) = 255.313989;
    
    //For test
    // K(0,0) = 1.0;
    // K(1,1) = 1.0;
    // K(2,2) = 1.0;

    cv::Matx33d E;
    E = K.t() * F * K;
    return E;
}

int relativepose(cv::Matx33d E, cv::Matx34d& P1, cv::Matx34d& P2, cv::Matx34d& P3, cv::Matx34d& P4){
    cv::SVD svd(E);
    cv::Mat U(svd.u);
    cv::Mat W(svd.w);
    cv::Mat VT(svd.vt);
    cv::Mat Wtmp(3,3,CV_64FC1, Scalar(0));
    Wtmp.at<double>(0,1) = -1.0;
    Wtmp.at<double>(1,0) = 1.0;
    Wtmp.at<double>(2,2) = 1.0;
    cv::Matx33d R1 = (cv::Mat)(U * Wtmp * VT);
    if(cv::determinant(R1)<0){
        R1=-R1;
    }
    cv::Matx31d T1,T2;
    cv::Matx33d R2 = (cv::Mat)(U * Wtmp.t() * VT);
    if(cv::determinant(R2)<0){
        R2=-R2;
    }
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++){
            P1(i,j) = R1(i,j);
            P2(i,j) = R2(i,j);
            P3(i,j) = R1(i,j);
            P4(i,j) = R2(i,j);
        }
    }
    for (int i = 0; i < 3; i++){
        P1(i,3) = U.at<double>(i,2);
        P2(i,3) = U.at<double>(i,2);
        T1(i) = U.at<double>(i,2);
        P3(i,3) = -U.at<double>(i,2);
        P4(i,3) = -U.at<double>(i,2);
        T2(i) = -U.at<double>(i,2);
    }
    return 0;
}

int triangulation(cv::Matx34d P1, cv::Matx34d P, vector<cv::Point2f> KPS_prev, vector<cv::Point2f> KPS_next){
    Matx14d P11,P12,P13;
    Matx14d p1,p2,p3;
    int count = 0;
    for (int i = 0; i < 4; i++){
        p1(i) = P(0,i);
        p2(i) = P(1,i);
        p3(i) = P(2,i);
        P11(i) = P1(0,i);
        P12(i) = P1(1,i);
        P13(i) = P1(2,i);
    }
    Matx44d P4_1, P4_2;
    for (int row = 0; row < 3; row++){
        for (int col = 0; col < 4; col++){
            P4_1(row,col) = P1(row,col);
            P4_2(row,col) = P(row,col);
        }
    }

    for (int j = 0; j < 3; j++){
        P4_1(3,j) = 0.0;
        P4_2(3,j) = 0.0;
    }
    P4_1(3,3) = 1.0; P4_2(3,3) = 1.0;s

    for (int i = 0; i < KPS_prev.size(); i++){
        Matx44d A;
        Matx14d a1 = KPS_prev[i].y * P13 - P12;
        Matx14d a2 = P11 - KPS_prev[i].x * P13;
        Matx14d a3 = KPS_next[i].y * p3 - p2;
        Matx14d a4 = p1 - KPS_next[i].x * p3;
        for (int i = 0; i < 4; i++){
            A(0,i) = a1(0,i);
            A(1,i) = a2(0,i);
            A(2,i) = a3(0,i);
            A(3,i) = a4(0,i);
        }
        cv::SVD svd(A);
        cv::Mat VT(svd.vt);
        cv::Mat Xtmp(VT.row(VT.rows-1));
        Matx41d X = (cv::Mat)(Xtmp.t());
        Matx41d X1 = P4_1 * X;
        X1(2) = X1(2)/X1(3);
        Matx41d X2 = P4_2 * X;
        X2(2) = X2(2)/X2(3);
        if (X1(2) > 0 && X2(2) >0){
            count++;
        }
    }
    cout << "count" << count << endl;
    return count;

}



 float distancePointLine(const cv::Point2f point, const cv::Vec<double,3>& line)
 {
   //Line is given as a*x + b*y + c = 0
   return std::fabs(line(0)*point.x + line(1)*point.y + line(2))
       / std::sqrt(line(0)*line(0)+line(1)*line(1));
 }

void drawEpipolarLines(const std::string& title, const cv::Matx33d F,
                 cv::Mat& img1, cv::Mat& img2,
                 std::vector<cv::Point2f> points1,
                 std::vector<cv::Point2f> points2,
                 const float inlierDistance = -1)
 {
    CV_Assert(img1.size() == img2.size() && img1.type() == img2.type());
    cv::Mat outImg1(img1.rows, img1.cols*2, CV_8UC1);
    cv::cvtColor(img1, outImg1, CV_BGR2GRAY);
    cv::Mat outImg2(img1.rows, img1.cols*2, CV_8UC1);
    cv::cvtColor(img2, outImg2, CV_BGR2GRAY);

    std::vector<cv::Vec3f> epilines1, epilines2;
    cv::computeCorrespondEpilines(points1, 1, F, epilines1); //Index starts with 1
    cv::computeCorrespondEpilines(points2, 2, F, epilines2);

    for(size_t i=0; i<points1.size(); i++){
        cv::Scalar color(255);
        int temp1 = img1.cols;
        int temp2 = img2.cols;
        cv::line(outImg1,
        cv::Point(0,-epilines1[i][2]/epilines1[i][1]),
        cv::Point(temp1,-(epilines1[i][2]+epilines1[i][0]*temp1)/epilines1[i][1]),
        color);
        cv::circle(outImg1, points1[i], 3, color, -1, CV_AA);

        cv::line(outImg2,
        cv::Point(0,-epilines2[i][2]/epilines2[i][1]),
        cv::Point(temp2,-(epilines2[i][2]+epilines2[i][0]*temp2)/epilines2[i][1]),
        color);
        cv::circle(outImg2, points2[i], 3, color, -1, CV_AA);
    }

    Mat HImg;
    hconcat(outImg1, outImg2, HImg);
    cv::imwrite( "../../epi.jpg", HImg );
    namedWindow(title, WINDOW_AUTOSIZE );
    cv::imshow(title, HImg);
    cv::waitKey(0);
 }

 void testresult(){
    cv::Mat points3D(1, 16, CV_64FC4);
    cv::randu(points3D, cv::Scalar(-5.0, -5.0, 1.0, 1.0), cv::Scalar(5.0, 5.0, 10.0, 1.0 ));


    //Compute 2 camera matrices
    cv::Matx34d C1 = cv::Matx34d::eye();
    cv::Matx34d C2 = cv::Matx34d::eye();
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    C2(2, 3) = 1;

    //Compute points projection
    vector<cv::Point2f> points1;
    vector<cv::Point2f> points2;

    for(size_t i = 0; i < points3D.cols; i++)
    {
        cv::Vec3d hpt1 = C1*points3D.at<cv::Vec4d>(0, i);
        cv::Vec3d hpt2 = C2*points3D.at<cv::Vec4d>(0, i);

        hpt1 /= hpt1[2];
        hpt2 /= hpt2[2];

        cv::Point2f p1(hpt1[0], hpt1[1]);
        cv::Point2f p2(hpt2[0], hpt2[1]);

        points1.push_back(p1);
        points2.push_back(p2);
    }
    cv::Matx34d cameraP;
    cameraP(0,0) = 1.0;
    cameraP(1,1) = 1.0;
    cameraP(2,2) = 1.0;
    cout << "cameraP" << cameraP << endl;
    Matx33d F = Findfundamental(points1,points2);
    cv::Matx33d Essential = Findessential(F);

    cv::Matx34d P[4]; // initialize 4 camera pose
    relativepose(Essential, P[0], P[1], P[2], P[3]);
    int num = 0;
    cv::Matx34d BestP;
    for (int i = 0; i < 4; i++){
        int num_tmp  = triangulation(cameraP, P[i], points1, points2);
        if (num_tmp > num){
            BestP = P[i];
            num = num_tmp;
        }   
    }

    // check if R and t are correct
    cv::Matx33d cameraK; // Output 3x3 camera matrix K.
    cv::Matx33d R_self; // Output 3x3 external rotation matrix R.
    cv::Matx31d t_self; //Output 4x1 translation vector T.
    cv::Matx34d Ptmp = BestP;
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++){
            R_self(i,j) = Ptmp(i,j);
        }
    }
    for (int i = 0; i < 3; i++){
        t_self(i) = Ptmp(i,3);
    } 
    cout << "R_self" << R_self << endl;
    cout << "t_self" << t_self << endl;


    cout << "C2_original" << C2 << endl;



 }
 


int main( int argc, char** argv )
{
    int method = 1;// 1-self;2-opencv;
    srand ( time(NULL) );

    if ( argc != 3 )
    {
        cout<<"usage: feature_extraction img1 img2"<<endl;
        return 1;
    }
    //-- Read two images
    Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );
    Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );

    list< cv::Point2f > keypoints;
    vector<cv::KeyPoint> kps;

    std::string detectorType = "Feature2D.BRISK";
    Ptr<FeatureDetector>detector = Algorithm::create<FeatureDetector>(detectorType);
    detector->set("thres", 100);


    detector->detect( img_1, kps );
    for ( auto kp:kps )
        keypoints.push_back( kp.pt );

    vector<cv::Point2f> next_keypoints;
    vector<cv::Point2f> prev_keypoints;
    for ( auto kp:keypoints )
        prev_keypoints.push_back(kp);
    vector<unsigned char> status;
    vector<float> error;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    cv::calcOpticalFlowPyrLK( img_1, img_2, prev_keypoints, next_keypoints, status, error );
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<"LK Flow use time："<<time_used.count()<<" seconds."<<endl;

    vector<cv::Point2f> kps_prev,kps_next;
    // vector<cv::Point2f> KPS_prev,KPS_next; // Draw a figure to show epipolar constraint
    kps_prev.clear();
    kps_next.clear();
    for(size_t i=0;i<prev_keypoints.size();i++)
    {
        if(status[i] == 1)
        {
            kps_prev.push_back(prev_keypoints[i]);
            kps_next.push_back(next_keypoints[i]);
            // KPS_prev.push_back(prev_keypoints[i]);
            // KPS_next.push_back(next_keypoints[i]);
        }
    }

    double p = 0.9;
    double d = 0.5f;
    double e = 0.2;

    // int niter = static_cast<int>(std::ceil(std::log(1.0-p)/std::log(1.0-std::pow(1.0-e,8))));
    int niter = 500;
    Mat Fundamental;
    cv::Matx33d F,Fcandidate;
    int bestinliers = -1;
    vector<cv::Point2f> prev_subset,next_subset;
    int matches = kps_prev.size();
    prev_subset.clear();
    next_subset.clear();
    // cout << "size: " << KPS_prev.size() << endl;

    for(int i=0;i<niter;i++){
        // step1: randomly sample 8 matches for 8pt algorithm
        unordered_set<int> rand_util;
        while(rand_util.size()<8)
        {
            int randi = rand() % matches;
            rand_util.insert(randi);
        }
        vector<int> random_indices (rand_util.begin(),rand_util.end());
        for(size_t j = 0;j<rand_util.size();j++){
            prev_subset.push_back(kps_prev[random_indices[j]]);
            next_subset.push_back(kps_next[random_indices[j]]);
        }
        // step2: perform 8pt algorithm, get candidate F
        if (method == 1){
            Fcandidate = Findfundamental(prev_subset,next_subset);
        }
        else{
            Fcandidate = (cv::Matx33d)cv::findFundamentalMat(prev_subset,next_subset, CV_FM_8POINT);
        }

        // step3: Evaluate inliers, decide if we need to update the best solution
        int inliers = 0;
        for(size_t j=0;j<kps_prev.size();j++){
            if(checkinlier(kps_prev[j],kps_next[j],Fcandidate,d))
                inliers++;
        }
        if(inliers > bestinliers)
        {
            F = Fcandidate;
            bestinliers = inliers;
            // cout << "bestinliers: " << bestinliers << endl;
        }
        prev_subset.clear();
        next_subset.clear();
    }

    // step4: After we finish all the iterations, use the inliers of the best model to compute Fundamental matrix again.

    for(size_t j=0;j<kps_prev.size();j++){
        if(checkinlier(kps_prev[j],kps_next[j],F,d))
        {
            prev_subset.push_back(kps_prev[j]);
            next_subset.push_back(kps_next[j]);
        }

    }
    if (method == 1){
        F = Findfundamental(prev_subset,next_subset);
    }
    else{
        F = (cv::Matx33d)cv::findFundamentalMat(prev_subset,next_subset, CV_FM_8POINT);
    } 
    
    if (method == 1){
        cout<<"Fundamental matrix is \n"<<F<<endl;
    }
    else{
        cout << "Fundmental matrix from opencv is \n" << F << endl;
    }

    cv::Matx33d Essential = Findessential(F);
    cv::Matx34d P[4]; // initialize 4 camera pose
    relativepose(Essential, P[0], P[1], P[2], P[3]);

    // compute camera P1
    cv::Matx34d cameraP;
    for (int i = 0; i < 3; i++){
        cameraP(i,i) = 1.0;
    }
    cv::Matx33d K;
    K(0,0) = -517.3;
    K(1,1) = -516.5;
    K(2,2) = 1.0;
    K(0,2) = 318.643040;
    K(1,2) = 255.313989;

    int num = 0;
    cv::Matx34d BestP;
    for (int i = 0; i < 4; i++){
        int num_tmp  = triangulation(K * cameraP, K * P[i], kps_prev,kps_next);
        if (num_tmp > num){
            BestP = P[i];
            num = num_tmp;
        }   
    }

    // check if R and t are correct
    cv::Matx33d R_self; // Output 3x3 external rotation matrix R.
    cv::Matx31d t_self; //Output 4x1 translation vector T.

    cv::Matx34d Ptmp = K.inv() * BestP;
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++){
            R_self(i,j) = Ptmp(i,j);
        }
    }
    for (int i = 0; i < 3; i++){
        t_self(i) = Ptmp(i,3);
    } 
    cout << "R_self" << R_self << endl;
    cout << "t_self" << t_self << endl;

    Locations of triangulated 3d map points
    cv::Matx41d X_prev, X_next;
    for (int i = 0; i < 2; i++){
        cv::Matx31d x_prev;
        cv::Matx31d x_next;
        x_prev(0) =  kps_prev[i].x; x_prev(1) =  kps_prev[i].y; x_prev(2) =  1.0;
        x_next(0) =  kps_prev[i].x; x_next(1) =  kps_prev[i].y; x_next(2) =  1.0;
        X_prev = BestP.inv() * x_prev;
        X_prev = X_prev / (double)X_prev(3);
        X_next = BestP.inv() * x_next;
        X_next = X_next / (double)X_next(3);
        cout << X_next.size() << endl;
        cout << "x_prev" << x_prev << endl;
        cout << "X_prev" << X_prev << endl;
        cout << "x_next" << x_next << endl;
        cout << "X_next" << X_next << endl;
        
    }
    
    // draw epoploarlines
    drawEpipolarLines("epipolar line", F, img_1, img_2, kps_prev, kps_next, -1);
    
    testresult();
    return 0;
}

