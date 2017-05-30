/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef FRAME_H
#define FRAME_H

#include<vector>

#include "MapPoint.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "KeyFrame.h"
#include "ORBextractor.h"

#include <opencv2/opencv.hpp>

namespace ORB_SLAM2
{
#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

class MapPoint;
class KeyFrame;

class Frame
{
public:
    Frame();

    // Copy constructor.
    Frame(const Frame &frame);

    // Constructor for stereo cameras.
    Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);

    // Constructor for RGB-D cameras.
    Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);

    // Constructor for Monocular cameras.
    Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);

    //提取图像的ORB描述子：0表示左图，1表示右图
    // ORB特征是将FAST特征点的检测方法与BRIEF特征描述子结合起来,并在它们原来的基础上做了改进与优化；使用orbExtractor()提取
    // mvkeys 和mvkeysRight 存双目keypoints位置(未矫正)
    // mDescriptors, mvDescriptorsRight 存各个keypoints对应的orb描述子
    void ExtractORB(int flag, const cv::Mat &im);

    // 计算当前帧的BoW,存放在mBowVec中
    void ComputeBoW();

    // 直接用指定的Tcw更新mTcw，用作相机位姿
    void SetPose(cv::Mat Tcw);

    // 用mTcw更新Rota,Trans和相机光心位置
    void UpdatePoseMatrices();

    // 返回光心位置
    inline cv::Mat GetCameraCenter()
	{
        return mOw.clone();
    }

    // Returns inverse of rotation; Inverse is the same as transpose;
    inline cv::Mat GetRotationInverse()
	{
        return mRwc.clone();
    }

    // 判断指定的MapPoint 是否在相机视野中
    // 完善该MapPoint 在tracking中将会被用到的参数
    // 在tracking 的SearchLocalPoints()中使用
    bool isInFrustum(MapPoint* pMP, float viewingCosLimit);

    // 计算指定的keypoint 在图像中所处的网格（posX，posY），在图像外则返回False
    // 只在Frame 的构造函数中使用
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

    // 找到参数指定区域的特征点（已矫正的），返回满足各个条件的特征点序号
    /// size_t: long unsigned int
    vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel=-1, const int maxLevel=-1) const;

    // 搜索左右图像匹配的keypoints，并求深度保存到mvDepth,保存右目keypoints坐标到mvuRight
    // mvDepth,mvuRight 与keypoints数目一样，但只有有匹配才被赋值
    void ComputeStereoMatches();

    // Associate a "right" coordinate to a keypoint if there is valid depth in the depthmap.
    void ComputeStereoFromRGBD(const cv::Mat &imDepth);

    // 反投影，将双目/深度图的keypoints投影到世界3D世界坐标系下
    cv::Mat UnprojectStereo(const int &i);

public:
    // Vocabulary used for relocalization; tracking中有同名变量，两者指向同一词汇表
    ORBVocabulary* mpORBvocabulary;

    // Feature extractor. The right is used only in the stereo case; tracking中有同名变量，指向对应的相同的提取器
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;

    // 当前Frame实例的时间戳
    double mTimeStamp;

    // Calibration matrix and OpenCV distortion parameters.
    cv::Mat mK;
    static float fx;
    static float fy;
    static float cx;
    static float cy;
    static float invfx;
    static float invfy;
    cv::Mat mDistCoef;

    // Stereo baseline multiplied by fx.
    float mbf;

    // Stereo baseline in meters.
    float mb;

    // 远近点判断门限. Close points are inserted from 1 view.
    // Far points are inserted as in the monocular case from 2 views.
    float mThDepth;

    // 当前实例的KeyPoints 数量.
    int N; ///< KeyPoints数量

    // mvKeys: 原始左图像提取出的Keypoints向量（未校正）
    // mvKeysRight: 原始右图像提取出的Keypoints向量（未校正）
    // mvKeysUn: mvKeys校正后的特征点，对于双目摄像头，一般得到的图像都是校正好的，再校正一次是冗余的；
    //           对于RGBD图像则需要矫正，因为一般都是未矫正的
    std::vector<cv::KeyPoint> mvKeys, mvKeysRight;
    std::vector<cv::KeyPoint> mvKeysUn;

    // mvuRight： 左右目匹配的Keypoints对中，对应于左目Keypoints的右目Keypoints在其图像坐标系中的x坐标
    // mvDepth: 左右目匹配的Keypoints对应的深度
    // 两个变量长度和左目Keypoints 数量一样
    // 没有匹配的keypoints的位置其内容为-1; 单目摄像头，这两个容器中存的都是-1
    std::vector<float> mvuRight;
    std::vector<float> mvDepth;

    // Bag of Words Vector structures.
    DBoW2::BowVector mBowVec;       // 虽然是一个类，但主要的是其中 std::map<wordId, wordsValue>, 存储的是本帧的Bow向量中word--weight的映射; wordId只有根节点才有
    DBoW2::FeatureVector mFeatVec;      // 虽然是一个类，但主要的是其中 std::map<NodeId, std::vector<unsigned int>>, 前者是词汇树的节点号，后者是节点聚类中心descriptor(orb)对应于本实例中feature的编号

    // 左目摄像头和右目摄像头特征点对应的描述子矩阵，每一行对应一个Keypoint
    cv::Mat mDescriptors, mDescriptorsRight;

    // 每一个Keypoint对应的MapPoint，指针向量
    std::vector<MapPoint*> mvpMapPoints;

    // 表明keypoints--mappoints 是否outlier的向量， 与keypoints数目相同
    std::vector<bool> mvbOutlier;

    // 将图像分成若干个网格，坐标乘以mfGridElementWidthInv和mfGridElementHeightInv就可以确定在哪个格子，便于匹配和搜索
    static float mfGridElementWidthInv;
    static float mfGridElementHeightInv;
    // FRAME_GRID_ROWS 48
    // FRAME_GRID_COLS 64
    // 二维数组，每个单元对应一个图像的网格，其中存储一个向量，向量中是该网格中所有特征点的序号（类似哈希表用于检索）
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    // Camera pose.
    cv::Mat mTcw; ///< 相机姿态 世界坐标系到相机坐标坐标系的变换矩阵

    // Current and Next Frame id.
    static long unsigned int nNextId; ///< Next Frame id.
    long unsigned int mnId; ///< Current Frame id.

    // Reference Keyframe.
    KeyFrame* mpReferenceKF;//指针，指向参考关键帧

    // Scale pyramid info.
    int mnScaleLevels;//图像提金字塔的层数
    float mfScaleFactor;//图像提金字塔的尺度因子
    float mfLogScaleFactor;//
    vector<float> mvScaleFactors;
    vector<float> mvInvScaleFactors;
    vector<float> mvLevelSigma2;
    vector<float> mvInvLevelSigma2;

    // 对矫正后的图像划分网格，求mfGridElementxxInv时的使用的图像边界参数
    static float mnMinX;
    static float mnMaxX;
    static float mnMinY;
    static float mnMaxY;

    // 初始化时标志是否进行网格划分和矫正（当矫正参数变化时也为true）———— 尚未用到?
    static bool mbInitialComputations;


private:

    // Undistort keypoints given OpenCV distortion parameters.
    // Only for the RGB-D case. Stereo must be already rectified!
    // (called in the constructor).
    void UndistortKeyPoints();

    // 在构造函数中使用，用于计算图像边界，即求解public中的mnMin/Max X/Y
    void ComputeImageBounds(const cv::Mat &imLeft);

    // 在构造函数中使用，根据keypoints位置将他们的序号分配到对应网格 mGrid[·][·]
    void AssignFeaturesToGrid();

    // Rotation, translation and camera center
    cv::Mat mRcw; ///< Rotation from world to camera
    cv::Mat mtcw; ///< Translation from world to camera
    cv::Mat mRwc; ///< Rotation from camera to world
    cv::Mat mOw;  ///< mtwc,Translation from camera to world
};

}// namespace ORB_SLAM

#endif // FRAME_H
