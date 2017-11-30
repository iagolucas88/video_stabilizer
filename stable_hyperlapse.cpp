#include <opencv2/opencv.hpp>
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>


using namespace std;
using namespace cv;

const int HORIZONTAL_BORDER_CROP = 20; 

struct TransformParam
{
    TransformParam() {}
    TransformParam(double _dx, double _dy, double _da) {
        dx = _dx;
        dy = _dy;
        da = _da;
    }

    double dx;
    double dy;
    double da; 
};

struct Trajectory
{
    Trajectory() {}
    Trajectory(double _x, double _y, double _a) {
        x = _x;
        y = _y;
        a = _a;
    }
    // "+"
    friend Trajectory operator+(const Trajectory &c1,const Trajectory  &c2){
        return Trajectory(c1.x+c2.x,c1.y+c2.y,c1.a+c2.a);
    }
    //"-"
    friend Trajectory operator-(const Trajectory &c1,const Trajectory  &c2){
        return Trajectory(c1.x-c2.x,c1.y-c2.y,c1.a-c2.a);
    }
    //"*"
    friend Trajectory operator*(const Trajectory &c1,const Trajectory  &c2){
        return Trajectory(c1.x*c2.x,c1.y*c2.y,c1.a*c2.a);
    }
    //"/"
    friend Trajectory operator/(const Trajectory &c1,const Trajectory  &c2){
        return Trajectory(c1.x/c2.x,c1.y/c2.y,c1.a/c2.a);
    }
    //"="
    Trajectory operator =(const Trajectory &rx){
        x = rx.x;
        y = rx.y;
        a = rx.a;
        return Trajectory(x,y,a);
    }

    double x;
    double y;
    double a; 
};
 

int main(int argc, char **argv)
{
    if(argc < 2) {
        cout << "Impossible to stabilize video for less than two frames!" << endl;
        return 0;
    }
    //Data for futhers analyses
    ofstream data_transform("data_transform.txt");
    ofstream data_trajectory("data_trajectory.txt");
    ofstream data_smoothed_trajectory("data_smoothed_trajectory.txt");
    ofstream data_new_transform("data_new_transformation.txt");

    VideoCapture cap(argv[1]);
    assert(cap.isOpened());

    Mat cur, cur_grey;
    Mat prev, prev_grey;

    cap >> prev;
    cvtColor(prev, prev_grey, COLOR_BGR2GRAY);
    
   
    vector <TransformParam> prev_to_cur_transform; 
    
    double a = 0;
    double x = 0;
    double y = 0;
    
    vector <Trajectory> trajectory;
    
    vector <Trajectory> smoothed_trajectory;
    Trajectory X;
    Trajectory X_;
    Trajectory P;
    Trajectory P_;
    Trajectory K;
    Trajectory z;
    double pstd = 4e-3;
    double cstd = 0.25;
    Trajectory Q(pstd,pstd,pstd);
    Trajectory R(cstd,cstd,cstd);
    vector <TransformParam> new_prev_to_cur_transform;


    Mat T(2,3,CV_64F);

    int vert_border = HORIZONTAL_BORDER_CROP * prev.rows / prev.cols; 
    int ex = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));
   
    char EXT[] = {(char)(ex & 0XFF) , (char)((ex & 0XFF00) >> 8),(char)((ex & 0XFF0000) >> 16),(char)((ex & 0XFF000000) >> 24), 0};
    cout << "Input codec type: " << EXT << endl; 
    VideoWriter outputVideo;
    outputVideo.open("stabilized.avi", CV_FOURCC('X','V','I','D'), cap.get(CV_CAP_PROP_FPS), cv::Size(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT)));


    if (!outputVideo.isOpened()){
        cout << "ERROR: Failed to write the video" << endl;
        return -1;
    }
      
    int k=1;
    int max_frames = cap.get(CV_CAP_PROP_FRAME_COUNT);
    Mat last_T;
    Mat prev_grey_,cur_grey_;

     
    while(true) {

        cap >> cur;
        if(cur.data == NULL) {
            break;
        }

        cvtColor(cur, cur_grey, COLOR_BGR2GRAY);

        vector <Point2f> prev_corner, cur_corner;
        vector <Point2f> prev_corner2, cur_corner2;
        vector <uchar> status;
        vector <float> err;

        goodFeaturesToTrack(prev_grey, prev_corner, 200, 0.01, 30);
        calcOpticalFlowPyrLK(prev_grey, cur_grey, prev_corner, cur_corner, status, err);

        for(size_t i=0; i < status.size(); i++) {
            if(status[i]) {
                prev_corner2.push_back(prev_corner[i]);
                cur_corner2.push_back(cur_corner[i]);
            }
        }

        Mat T = estimateRigidTransform(prev_corner2, cur_corner2, false);

        if(T.data == NULL) {
            last_T.copyTo(T);
        }

        T.copyTo(last_T);

        double dx = T.at<double>(0,2);
        double dy = T.at<double>(1,2);
        double da = atan2(T.at<double>(1,0), T.at<double>(0,0));

        data_transform << k << " " << dx << " " << dy << " " << da << endl;
       
        x += dx;
        y += dy;
        a += da;
        
        data_trajectory << k << " " << x << " " << y << " " << a << endl;
        
        z = Trajectory(x,y,a);
        
        if(k==1){
            
            X = Trajectory(0,0,0);
            P = Trajectory(1,1,1); 
        }
        else
        {
            
            X_ = X; 
            P_ = P+Q;
            K = P_/( P_+R ); 
            X = X_+K*(z-X_);  
            P = (Trajectory(1,1,1)-K)*P_; 
        }
 
        data_smoothed_trajectory << k << " " << X.x << " " << X.y << " " << X.a << endl;
        
        double diff_x = X.x - x;
        double diff_y = X.y - y;
        double diff_a = X.a - a;

        dx = dx + diff_x;
        dy = dy + diff_y;
        da = da + diff_a;

        data_new_transform << k << " " << dx << " " << dy << " " << da << endl;
        
        T.at<double>(0,0) = cos(da);
        T.at<double>(0,1) = -sin(da);
        T.at<double>(1,0) = sin(da);
        T.at<double>(1,1) = cos(da);

        T.at<double>(0,2) = dx;
        T.at<double>(1,2) = dy;

        Mat cur2;
        
        warpAffine(prev, cur2, T, cur.size());

        cur2 = cur2(Range(vert_border, cur2.rows-vert_border), Range(HORIZONTAL_BORDER_CROP, cur2.cols-HORIZONTAL_BORDER_CROP));
        outputVideo.write(cur2); 
        imshow("Stabilized", cur2);
        
        

        
        resize(cur2, cur2, cur.size());

        Mat canvas = Mat::zeros(cur.rows, cur.cols*2+10, cur.type());
        Mat separate_bar(cur.rows, 10, CV_8UC3, Scalar(0,255,0));

        prev.copyTo(canvas(Range::all(), Range(0, cur2.cols)));
        separate_bar.copyTo(canvas(Range::all(), Range(cur2.cols, cur2.cols+10)));
        cur2.copyTo(canvas(Range::all(), Range(cur2.cols+10, cur2.cols*2+10)));

        
        if(canvas.cols > 1920) {
            resize(canvas, canvas, Size(canvas.cols/2, canvas.rows/2));
        }

        imshow("Before | After", canvas);
        

        waitKey(10);
        //
        prev = cur.clone();
        cur_grey.copyTo(prev_grey);

        cout << "Optical Flow: " << prev_corner2.size() << " (" << k << "/" << max_frames << ")" << endl;
        k++;

    }
    outputVideo.release();
    return 0;
}