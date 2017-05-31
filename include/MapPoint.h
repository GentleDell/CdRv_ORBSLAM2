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

#ifndef MAPPOINT_H
#define MAPPOINT_H

#include"KeyFrame.h"
#include"Frame.h"
#include"Map.h"

#include<opencv2/core/core.hpp>
#include<mutex>

namespace ORB_SLAM2
{

class KeyFrame;
class Map;
class Frame;


class MapPoint
{
public:
    MapPoint(const cv::Mat &Pos, KeyFrame* pRefKF, Map* pMap);      // 给定位姿，KF,当前Map,构建MapPoints ———— initial时使用
    MapPoint(const cv::Mat &Pos,  Map* pMap, Frame* pFrame, const int &idxF);       // 给定位姿，当前Map，Frame，以及Mappoint在Frame中的索引，构建MapPoints ———— 双目，updateLastFrame()使用

    void SetWorldPos(const cv::Mat &Pos);
    cv::Mat GetWorldPos();

    cv::Mat GetNormal();        // 获得本实例的平均观测方向
    KeyFrame* GetReferenceKeyFrame();       // 获得本实例的参考关键帧  ？

    std::map<KeyFrame*,size_t> GetObservations();       // 获得观测到本实例的KF,以及相应keypoints的索引
    int Observations();     // 获取本实例被观测到的次数 nObs

    void AddObservation(KeyFrame* pKF,size_t idx);      // 记录观测到本实例的指定的KF以及相应keypoints的索引到mObservations,增加观测数 nObs
    void EraseObservation(KeyFrame* pKF);       // 擦除指定KF对本实例的观测记录，若本实例观测数少，将被清除

    int GetIndexInKeyFrame(KeyFrame* pKF);      // 返回int，是本实例在指定KF中对应的KeyPoint的序号
    bool IsInKeyFrame(KeyFrame* pKF);

    void SetBadFlag();      // 置mbBad=True，并将所有观测到本实例的KF对本实例的观测记录删除，擦除MP的内存
    bool isBad();       // 返回mbBad

    void Replace(MapPoint* pMP);
    MapPoint* GetReplaced();        // 返回替代了本实例的MP————mpReplaced (Loop中替代)

    void IncreaseVisible(int n=1);      // 增加可见数的记录， mnvisible += n
    void IncreaseFound(int n=1);        // 增加可以找到本实例的记录 Replace与Tracking中会使用
    float GetFoundRatio();
    inline int GetFound(){
        return mnFound;
    }

    void ComputeDistinctiveDescriptors();

    cv::Mat GetDescriptor();

    void UpdateNormalAndDepth();

    float GetMinDistanceInvariance();
    float GetMaxDistanceInvariance();
    int PredictScale(const float &currentDist, KeyFrame*pKF);       // 根据给定深度和KeyFrame，估计对应KeyPoint在金字塔的层数
    int PredictScale(const float &currentDist, Frame* pF);

public:
    long unsigned int mnId;     // 本实例的GlobalId ———— 所有创建过的MP中的Id
    static long unsigned int nNextId;
    long int mnFirstKFid;       // 建立本实例的KF的Id ———— 第一个构建函数
    long int mnFirstFrame;      // 建立本实例的FrameId ———— 第二个构建函数
    int nObs;       // 本实例被观测的次数

    // Variables used by the tracking
    float mTrackProjX;
    float mTrackProjY;
    float mTrackProjXR;
    bool mbTrackInView;
    int mnTrackScaleLevel;
    float mTrackViewCos;
    long unsigned int mnTrackReferenceForFrame;     // 防止将本实例重复添加到mvpLocalMap的标记（记录FrameId）
    long unsigned int mnLastFrameSeen;

    // Variables used by local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnFuseCandidateForKF;

    // Variables used by loop closing
    long unsigned int mnLoopPointForKF;
    long unsigned int mnCorrectedByKF;
    long unsigned int mnCorrectedReference;    
    cv::Mat mPosGBA;
    long unsigned int mnBAGlobalForKF;


    static std::mutex mGlobalMutex;

protected:    

     // Position in absolute coordinates
     cv::Mat mWorldPos;

     // Keyframes observing the point and associated index in keyframe
     std::map<KeyFrame*,size_t> mObservations;

     // Mean viewing direction
     cv::Mat mNormalVector;

     // Best descriptor to fast matching
     cv::Mat mDescriptor;

     // Reference KeyFrame
     KeyFrame* mpRefKF;

     // Tracking counters
     int mnVisible;
     int mnFound;

     // Bad flag (we do not currently erase MapPoint from memory)
     bool mbBad;
     MapPoint* mpReplaced;

     // Scale invariance distances
     float mfMinDistance;
     float mfMaxDistance;

     Map* mpMap;

     std::mutex mMutexPos;
     std::mutex mMutexFeatures;
};

} //namespace ORB_SLAM

#endif // MAPPOINT_H
