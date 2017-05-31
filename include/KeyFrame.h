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

#ifndef KEYFRAME_H
#define KEYFRAME_H

#include "MapPoint.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "ORBextractor.h"
#include "Frame.h"
#include "KeyFrameDatabase.h"

#include <mutex>


namespace ORB_SLAM2
{

class Map;
class MapPoint;
class Frame;
class KeyFrameDatabase;

class KeyFrame
{
public:
    KeyFrame(Frame &F, Map* pMap, KeyFrameDatabase* pKFDB);     // 设置实例中的标志和采纳数，将Frame中的Grid复制一份(vector<vector<vector>>)

    // Pose functions
    void SetPose(const cv::Mat &Tcw);
    cv::Mat GetPose();
    cv::Mat GetPoseInverse();
    cv::Mat GetCameraCenter();
    cv::Mat GetStereoCenter();
    cv::Mat GetRotation();
    cv::Mat GetTranslation();

    // 计算本实例的BoW, 保存在BoWVec 和FeatureVec（FeatureVec选择在第4层，一种可能是为了提高搜索效率，层数越多指数增加）
    void ComputeBoW();

    // Covisibility graph functions
    void AddConnection(KeyFrame* pKF, const int &weight);       // 为本实例与指定KeyFrame之间添加连接与权值（ map<KeyFrame*, in> ）,用于建立共视关键帧间的联系
    void EraseConnection(KeyFrame* pKF);
    void UpdateConnections();       // 更新本实例的KF连接，形成convisibility graph
    void UpdateBestCovisibles();        // 在Add与Erase函数中被调用，对连接到本实例的共视KF按连接权值进行排序
    std::set<KeyFrame *> GetConnectedKeyFrames();       // 返回与本实例连接的所有KF的指针
    std::vector<KeyFrame* > GetVectorCovisibleKeyFrames();
    std::vector<KeyFrame*> GetBestCovisibilityKeyFrames(const int &N);  // 返回按权重排序前N个的共视关键帧
    std::vector<KeyFrame*> GetCovisiblesByWeight(const int &w);     // 返回权重在 w 以上的共视关键帧
    int GetWeight(KeyFrame* pKF);

    // Spanning tree functions
    void AddChild(KeyFrame* pKF);
    void EraseChild(KeyFrame* pKF);
    void ChangeParent(KeyFrame* pKF);
    std::set<KeyFrame*> GetChilds();
    KeyFrame* GetParent();
    bool hasChild(KeyFrame* pKF);

    // Loop Edges
    void AddLoopEdge(KeyFrame* pKF);
    std::set<KeyFrame*> GetLoopEdges();

    // MapPoint observation functions
    void AddMapPoint(MapPoint* pMP, const size_t &idx);
    void EraseMapPointMatch(const size_t &idx);
    void EraseMapPointMatch(MapPoint* pMP);
    void ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP);
    std::set<MapPoint*> GetMapPoints();
    std::vector<MapPoint*> GetMapPointMatches();    // 获取与本实例中keypoint关联的MapPoints
    int TrackedMapPoints(const int &minObs);        // 返回本实例中高质量的MapPoints数量; 高质量指被多于指定的minObs个关键帧看到
    MapPoint* GetMapPoint(const size_t &idx);       // 与上第二行相似，但会滤除不良的点以及没有match的点

    // KeyPoint functions
    std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r) const;      // 获取指定位置区域的keypoints
    cv::Mat UnprojectStereo(int i);     // 将指定的第i个Keypoints反投影到3D中

    // Image
    bool IsInImage(const float &x, const float &y) const;

    // Enable/Disable bad flag changes
    void SetNotErase();
    void SetErase();

    // Set/check bad flag
    void SetBadFlag();
    bool isBad();

    // Compute Scene Depth (q=2 median). Used in monocular.
    float ComputeSceneMedianDepth(const int q);

    static bool weightComp( int a, int b){
        return a>b;
    }

    static bool lId(KeyFrame* pKF1, KeyFrame* pKF2){
        return pKF1->mnId<pKF2->mnId;
    }


    // The following variables are accesed from only 1 thread or never change (no mutex needed).
public:

    static long unsigned int nNextId;
    long unsigned int mnId;
    const long unsigned int mnFrameId;      // 记录本实例由哪一个Frame初始化

    const double mTimeStamp;

    // Grid (to speed up feature matching)
    const int mnGridCols;
    const int mnGridRows;
    const float mfGridElementWidthInv;
    const float mfGridElementHeightInv;

    // Variables used by the tracking
    long unsigned int mnTrackReferenceForFrame;     // 在Tracking中使用，如果本实例被加入到某一CurrentFrame 的LocalKeyFrame，则本变量记录CurrentFrameID
    long unsigned int mnFuseTargetForKF;        // 标记本实例在LocalMapping的融合MapPoints步骤已经被使用

    // Variables used by the local mapping， 取值为零或者本实例的mnId时分别表示在LocalBA中是否参与优化并被修正
    long unsigned int mnBALocalForKF;   // 非零则本实例参加优化？
    long unsigned int mnBAFixedForKF;   // 非零则本实例不参加优化？

    // Variables used by the keyframe database
    long unsigned int mnLoopQuery;      // 将本实例标记为Loop候选帧的 正在进行LoopClosing的帧的编号
    int mnLoopWords;        // 记录本实例与其他某一KF共同的words个数，DetectLoopCandidates（）
    float mLoopScore;       // 记录本实例在DetectLoopCandidate()中的得分
    long unsigned int mnRelocQuery; // 以下三变量与上述三个变量含义相似
    int mnRelocWords;
    float mRelocScore;

    // Variables used by loop closing
    cv::Mat mTcwGBA;
    cv::Mat mTcwBefGBA;
    long unsigned int mnBAGlobalForKF;

    // Calibration parameters
    const float fx, fy, cx, cy, invfx, invfy, mbf, mb, mThDepth;

    // Number of KeyPoints
    const int N;

    // KeyPoints, stereo coordinate and descriptors (all associated by an index)
    const std::vector<cv::KeyPoint> mvKeys;
    const std::vector<cv::KeyPoint> mvKeysUn;
    const std::vector<float> mvuRight; // negative value for monocular points
    const std::vector<float> mvDepth; // negative value for monocular points
    const cv::Mat mDescriptors;

    //BoW
    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;

    // 相对于亲节点的位姿，badflag为真时才计算
    cv::Mat mTcp;

    // Scale
    const int mnScaleLevels;
    const float mfScaleFactor;
    const float mfLogScaleFactor;
    const std::vector<float> mvScaleFactors;
    const std::vector<float> mvLevelSigma2;
    const std::vector<float> mvInvLevelSigma2;

    // Image bounds and calibration
    const int mnMinX;
    const int mnMinY;
    const int mnMaxX;
    const int mnMaxY;
    const cv::Mat mK;


    // The following variables need to be accessed trough a mutex to be thread safe.
protected:

    // SE3 Pose and camera center
    cv::Mat Tcw;
    cv::Mat Twc;
    cv::Mat Ow;

    cv::Mat Cw; // Stereo middel point. Only for visualization

    // MapPoints associated to keypoints
    std::vector<MapPoint*> mvpMapPoints;

    // BoW
    KeyFrameDatabase* mpKeyFrameDB;     // 指向KeyFrameDataBase实例的指针，KFDB包含所有KF,检测Loop,Reloc以及词典
    ORBVocabulary* mpORBvocabulary;     // ORB词汇表指针，与上一类中包含的词典相同

    // Grid over the image to speed up feature matching
    std::vector< std::vector <std::vector<size_t> > > mGrid;

    std::map<KeyFrame*,int> mConnectedKeyFrameWeights;      // 与本实例连接的KF以及权重
    std::vector<KeyFrame*> mvpOrderedConnectedKeyFrames;        // 按权重排序的KF序列
    std::vector<int> mvOrderedWeights;

    // Spanning Tree and Loop Edges
    bool mbFirstConnection;     // 更新生成树时使用，完成双向连接后置为false
    KeyFrame* mpParent;
    std::set<KeyFrame*> mspChildrens;
    std::set<KeyFrame*> mspLoopEdges;

    // Bad flags
    bool mbNotErase;
    bool mbToBeErased;
    bool mbBad;    

    float mHalfBaseline; // Only for visualization

    Map* mpMap;

    std::mutex mMutexPose;
    std::mutex mMutexConnections;
    std::mutex mMutexFeatures;
};

} //namespace ORB_SLAM

#endif // KEYFRAME_H
