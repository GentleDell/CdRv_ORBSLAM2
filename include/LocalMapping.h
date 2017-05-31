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

#ifndef LOCALMAPPING_H
#define LOCALMAPPING_H

#include "KeyFrame.h"
#include "Map.h"
#include "LoopClosing.h"
#include "Tracking.h"
#include "KeyFrameDatabase.h"

#include <mutex>


namespace ORB_SLAM2
{

class Tracking;
class LoopClosing;
class Map;

class LocalMapping
{
public:
    LocalMapping(Map* pMap, const float bMonocular);

    // 指向其他进程的指针
    void SetLoopCloser(LoopClosing* pLoopCloser);

    void SetTracker(Tracking* pTracker);

    // 主体程序，进行建图
    void Run();

    void InsertKeyFrame(KeyFrame* pKF);     // 被Tracking调用，向队列mlNewKeysFrames中插入新的KF

    // Thread Synch
    void RequestStop();     // 请求停止局部建图；mbStopRequested，mbAbortBA = true
    void RequestReset();    // 请求重启建图；mbResetRequested = true
    bool Stop();        // 查看Stop需求与许可，标记; mbStopRequested && !mbNotStop 则赋值 mbstopped = true
    void Release();     // 释放局部建图，将所有停止标志置为false，删除之前存储的KeyFrames
    bool isStopped();       // 检查建图是否停止，返回mbstopped
    bool stopRequested();       // 查看stop的申请情况，返回mbStopRequested
    bool AcceptKeyFrames();     // 检查是否接受KeyFrames,返回mbAcceptKeyFrame
    void SetAcceptKeyFrames(bool flag);     // 根据输入bool值设置是否接受KF,设置mbAcceptKeyFrame
    bool SetNotStop(bool flag);     // 设置建图停止的许可标志mbNotStop，若不许可建图停止但建图已经停止则返回false表示设置失败，其他情况正常设置，返回true
                                    /// Tracking 的CreateNewKeyFrame()中有调用SetNotStop()函数，因此当LocalMapping停止时将不插入新的KF

    void InterruptBA();     // 设置BA打断标志，mbAbortBA = true

    void RequestFinish();       // 申请结束建图，mbFinishRequested = true; 在system::shutdown中调用
    bool isFinished();      //  检查建图是否结束，返回mbFinished

    int KeyframesInQueue(){
        unique_lock<std::mutex> lock(mMutexNewKFs);
        return mlNewKeyFrames.size();
    }

protected:

    bool CheckNewKeyFrames();       // 查看mlNewKeyFrames中是否有等待插入到地图的KeyFrame
    void ProcessNewKeyFrame();      // 取出mlNewKeyFrames中最老的一帧KF，将其与MapPoints进行关联，更新MP信息，更新共视图和 essential graph
    void CreateNewMapPoints();      // 用与当前KeyFrame共视程度较高的KeyFrames恢复一些MP

    void MapPointCulling();     // 剔除上方两个函数引入的不良MP
    void SearchInNeighbors();       // 融合当前KeyFrame与相邻KeyFrames间重复的MapPoints(二级相邻)，（Tracking里也会更新）

    void KeyFrameCulling();     // 剔除90%以上的MapPoints都可以被其他至少3个KeyFrames观测到的KeyFrames

    cv::Mat ComputeF12(KeyFrame* &pKF1, KeyFrame* &pKF2);       // 由两个KeyFrmaes的姿态计算两者间的基本矩阵 F = inv(K')*E*inv(K)

    cv::Mat SkewSymmetricMatrix(const cv::Mat &v);      // 由T矩阵获得S矩阵，向量叉乘转换为矩阵点乘

    bool mbMonocular;       // 传感器是单目的标志

    void ResetIfRequested();    // 如果mbResetRequested为真，即有Reset的申请，则清空KF队列和MP队列 (Run()中相应Tracking)
    bool mbResetRequested;      // Reset申请标志
    std::mutex mMutexReset;

    bool CheckFinish();     // 返回mbFinishRequested
    void SetFinish();       // 设置mbFinished和mbstopped为true; 当Run()中的while(1)结束时即运行
    bool mbFinishRequested;     // Fnish申请标志
    bool mbFinished;        // Finish状态标志
    std::mutex mMutexFinish;

    Map* mpMap;

    LoopClosing* mpLoopCloser;
    Tracking* mpTracker;

    std::list<KeyFrame*> mlNewKeyFrames;        // 待插入的KeyFrames的指针列表，

    KeyFrame* mpCurrentKeyFrame;        // 当前LocalMapping使用的KeyFrame,从上方这个列表中取出

    std::list<MapPoint*> mlpRecentAddedMapPoints;   // 待检测的MapPoints队列，会被Culling检测;来自于ProcessNewKeyFrame()和CreateNewMapPoints()

    std::mutex mMutexNewKFs;

    bool mbAbortBA;     // 标志BA状态

    bool mbStopped;     // 标志Local Mapping的stop状态
    bool mbStopRequested;       // 标志Stop申请状态
    bool mbNotStop;     // 建图停止的许可标志；在stop()中只有允许stop(Notstop = false)才stop
    std::mutex mMutexStop;

    bool mbAcceptKeyFrames;     // 标记LocalMapping是否接受KF
    std::mutex mMutexAccept;
};

} //namespace ORB_SLAM

#endif // LOCALMAPPING_H
