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


#include "Tracking.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"ORBmatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Map.h"
#include"Initializer.h"

#include"Optimizer.h"
#include"PnPsolver.h"

#include<iostream>

#include<mutex>


using namespace std;

namespace ORB_SLAM2
{

Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor):
    mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
    mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), mpSystem(pSys), mpViewer(NULL),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0)
{   // 上方初始化一些系統狀態量，功能標志，指針等
    // 導入相機參數

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);   // 当保存浮点数据或XML/YML文件;   XML和YAML可以嵌套的两种集合类型：映射（mappings）、序列（sequences）
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);   // 內參矩陣
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);   // 校準矩陣
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];    // 對於00-02: fps = 10
    if(fps==0)
        fps=30;

    // 用於檢查是否進行重定位以及是否插入關鍵幀的參數Max/Min Frames
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"]; // RGB通道順序
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // 讀入ORB的參數用於各自的初始化

    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if(sensor==System::STEREO || sensor==System::RGBD)
    {
        mThDepth = mbf*(float)fSettings["ThDepth"]/fx;      // 遠近點門限
        cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if(sensor==System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}


cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
{
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;
    // 彩色图转灰度图
    if(mImGray.channels()==3)
    {
        if(mbRGB)   // 確定通道顺序RGB
        {
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
        }
        else    // 通道顺序BGR
        {
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
        }
    }
    // 向Frame構造函數传入双目灰度图，时间戳，双目ORB描述子，ORB词汇表，內參，校准矩阵，以及用于判断点所处距离“远近”的门限等
    // 生成當前幀的Frame類實例
    mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}


cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}


cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp)
{
    mImGray = im;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
        mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
    else
        mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}

void Tracking::Track()
{
    if(mState==NO_IMAGES_YET)   // 更新系統跟蹤狀態
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState=mState;     // 將此前跟蹤狀態計爲上一時刻跟蹤狀態，新的時刻到了

    // Get Map Mutex -> Map cannot be changed   停止地图更新， 鎖整個函數
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    if(mState==NOT_INITIALIZED)         // track未初始化则进行初始化，成功初始化后 mState=OK
    {
        if(mSensor==System::STEREO || mSensor==System::RGBD)
            StereoInitialization();
        else
            MonocularInitialization();

        mpFrameDrawer->Update(this);    //

        if(mState!=OK)  // 如果初始化還是失敗就返回 ———— 當前幀Keypoint少於500個
            return;
    }
    else
    {
        // 系统已经初始化. 可以Track Frame.  (現在只有OK和LOST)
        bool bOK;

        // 利用motion model， KeyFrame 或 relocalization（如果lost） 初步估計相机位姿
        if(!mbOnlyTracking)              // 如果建图功能打开
        {
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.

            if(mState==OK)      // 跟踪状态良好
            {
                // Local Mapping might have changed some MapPoints tracked in last frame   更新那些change   ？
                CheckReplacedInLastFrame();

                if(mVelocity.empty() || mCurrentFrame.mnId<mnLastRelocFrameId+2)    // 如果还没有运动模型 或者 当前帧与重定位帧的id差2
                {
                    bOK = TrackReferenceKeyFrame();     // Keyframe match的结果；點数大于10个返回1,否則返回0
                }
                else
                {
                    bOK = TrackWithMotionModel();       // 使用运动模型track的bool结果，是否>10
                    if(!bOK)
                        bOK = TrackReferenceKeyFrame();     // 如果运动模型匹配点少于10个，就还是用keyframe試一下
                }
            }
            else    // 跟踪Lost了
            {
                bOK = Relocalization();         // 进行relocalization     ？
            }
        }
        else        // 仅仅进行tracking，不建图
        {
            // Localization Mode: Local Mapping is deactivated

            if(mState==LOST)
            {
                bOK = Relocalization();
            }
            else
            {
                if(!mbVO)       // 爲False 則說明匹配上的點很多，可以直接更新
                {


                    if(!mVelocity.empty())      // motion model 可用
                    {
                        bOK = TrackWithMotionModel();   // 這裏不會檢查效果然後換用 Ref KeyFrame
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else    // mbVO 为Ture，说明跟上的点太少，更新不了；
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    bool bOKMM = false;     // bOK Motion Model
                    bool bOKReloc = false;  // bOK relocalization
                    vector<MapPoint*> vpMPsMM;
                    vector<bool> vbOutMM;
                    cv::Mat TcwMM;
                    if(!mVelocity.empty())
                    {
                        bOKMM = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.mTcw.clone();
                    }
                    bOKReloc = Relocalization();

                    if(bOKMM && !bOKReloc)      // Motion Model 成功而 Relocalization失败，沿用MM的
                    {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;
                        //
                        if(mbVO)
                        {
                            for(int i =0; i<mCurrentFrame.N; i++)
                            {
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();     // 跟踪计数器mnFound + 1 ？   0.25f
                                }
                            }
                        }
                    }
                    else if(bOKReloc)   // Relocalization成功
                    {
                        mbVO = false;   // 這個時候一定有很多点关联到map point
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }
        }

        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // 如果有相机位姿的初始估计以及匹配结果，开始track local map.
        if(!mbOnlyTracking)     // 开启了建图
        {
            if(bOK)     // 此前track上了
                bOK = TrackLocalMap();      // 进行局部地图的跟踪
        }
        else
        {
            // 如果mbVO 为Ture，说明跟上的点太少，更新不了; 只有mbVO 为为False时才可以用局部地图
            // 等重定位后mbVO会被置为False，那时可以再次使用局部地图。
            if(bOK && !mbVO)
                bOK = TrackLocalMap();
        }

        if(bOK)     // 對Track的狀態進行更新
            mState = OK;
        else
            mState=LOST;

        // Update drawer
        mpFrameDrawer->Update(this);

        // 跟踪状态良好的话, 考虑插入做keyframe
        if(bOK)
        {
            // 更新运动模型
            if(!mLastFrame.mTcw.empty())    // 如果上一帧求得了位姿， 定位第一幀就進不來
            {
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                mVelocity = mCurrentFrame.mTcw*LastTwc;
            }
            else
                mVelocity = cv::Mat();  // 否则为空

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);      // 直接将输入pose拷贝到mCameraPose

            // 清除 VO 匹配
            for(int i=0; i<mCurrentFrame.N; i++)    // 对每一个 CurrentFrame的 Key point
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];      // 取出对应到该 Keypoint的 Mappoint
                if(pMP)     // 如果有对应的Map point
                    if(pMP->Observations()<1)       // 如果nObs<1； 初始化時會賦初值1或2的，小於1說明都被擦除了；或者是用於輔助匹配的點，本來就要刪除
                    {
                        mCurrentFrame.mvbOutlier[i] = false;    // 由於馬上會被置空指針，就標記爲非outlier
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);     // 赋空指针
                    }
            }

            // 删除临时MapPoints.   Delete temporal MapPoints
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;     // 刪除指針指向的內存 也可以 delete *lit
            }
            mlpTemporalPoints.clear();  // 清空mlpTemporalPoints這個list

            // 检查是否需要插入关键帧
            if(NeedNewKeyFrame())
                CreateNewKeyFrame();    // 需要的话生成关键帧

            // We allow points with high innovation (considererd outliers by the Huber Function)    即便根據Huber Function是outlier 的點我們也加入到新KeyFrame中
            // pass to the new keyframe, so that bundle adjustment will finally decide              讓BA決定它們到底是不是outlier。但位姿估計中不使用這些點，因此在本幀
            // if they are outliers or not. We don't want next frame to estimate its position       中忽略
            // with those points so we discard them in the frame.
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])    // 刪除outlier
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);     // CurrentFrame中不使用Outlier
            }
        }

        // 如果初始化不久就lost了, Reset
        if(mState==LOST)
        {
            if(mpMap->KeyFramesInMap()<=5)
            {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();
                return;
            }
        }

        if(!mCurrentFrame.mpReferenceKF)    // 当前帧的参考关键帧为空
            mCurrentFrame.mpReferenceKF = mpReferenceKF;    // 對應到Ref KeyFrame  之前復制可能爲空或者被刪掉

        mLastFrame = Frame(mCurrentFrame);  // 将当前帧复制到mLastFrame中——當前幀終於也成爲了歷史
    }

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    if(!mCurrentFrame.mTcw.empty())
    {
        cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();     // Tcr = mTcw * Twr 後者是RefFrame的Twc
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState==LOST);        // 邏輯判斷
    }
    else
    {
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());    // 復制
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }

}


void Tracking::StereoInitialization()
{   // 只有當前幀Keypoints 大於500點才初始化
    if(mCurrentFrame.N>500)
    {
        // 初始化位姿矩陣（Rotation, translation and camera center），置爲單位陣
        mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));    // eye(row, col, inttype)

        // 生成關鍵幀
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

        // 將生成的關鍵幀加入到Map中 —— 將初始幀作爲第一個關鍵幀
        mpMap->AddKeyFrame(pKFini);

        // 生成MapPoints 並與本關鍵幀關聯
        for(int i=0; i<mCurrentFrame.N;i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)     //對於深度有效的點
            {
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);     // Keypoint投影到3D 空間
                MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpMap);  // 以此新建MapPoint 並建立關聯
                pNewMP->AddObservation(pKFini,i);       // 建立本KeyFrame與本mappoint的觀察關系
                pKFini->AddMapPoint(pNewMP,i);      // 建立該Keypoint與Mappoint的聯系
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();     // 更新法向量和深度
                mpMap->AddMapPoint(pNewMP);     // 正式將Mappoint加入地圖

                mCurrentFrame.mvpMapPoints[i]=pNewMP;   // 建立CurrentFrame與本Mappoint的聯系
            }
        }

        cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;

        mpLocalMapper->InsertKeyFrame(pKFini);  // 將新建KeyFrame加入mlNewKeyFrames新建關鍵幀list中

        mLastFrame = Frame(mCurrentFrame);  // 當前幀復制爲舊幀
        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;    // 新建關鍵幀作爲上一關鍵幀

        mvpLocalKeyFrames.push_back(pKFini);    // 將新建幀包含到本地關鍵幀list中
        mvpLocalMapPoints=mpMap->GetAllMapPoints(); // 將所有當前新建的Mappoint都加入mvpLocalMapPoints，因爲這些是新點，還未關聯到新到Frame
        mpReferenceKF = pKFini;     // 生成的關鍵幀作爲Ref KeyFrame(文中提到)
        mCurrentFrame.mpReferenceKF = pKFini;   // 同上，但是位於Frame.h 作用 ？

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);    // 設置參考Mappoint ？

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);    // 設置初始關鍵幀  ？

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        mState=OK;  // Track狀態計爲OK
    }
}

void Tracking::MonocularInitialization()
{

    if(!mpInitializer)
    {
        // Set Reference Frame
        if(mCurrentFrame.mvKeys.size()>100)
        {
            mInitialFrame = Frame(mCurrentFrame);
            mLastFrame = Frame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            if(mpInitializer)
                delete mpInitializer;

            mpInitializer =  new Initializer(mCurrentFrame,1.0,200);

            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);

            return;
        }
    }
    else
    {
        // Try to initialize
        if((int)mCurrentFrame.mvKeys.size()<=100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            return;
        }

        // Find correspondences
        ORBmatcher matcher(0.9,true);
        int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);

        // Check if there are enough correspondences
        if(nmatches<100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            return;
        }

        cv::Mat Rcw; // Current Camera Rotation
        cv::Mat tcw; // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

        if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
        {
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }

            // Set Frame Poses
            mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
            Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(Tcw.rowRange(0,3).col(3));
            mCurrentFrame.SetPose(Tcw);

            CreateInitialMapMonocular();
        }
    }
}

void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpMap,mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);


    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i]<0)
            continue;

        //Create MapPoint.
        cv::Mat worldPos(mvIniP3D[i]);

        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpMap);

        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);

        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        mpMap->AddMapPoint(pMP);
    }

    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

    Optimizer::GlobalBundleAdjustemnt(mpMap,20);

    // Set median depth to 1
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f/medianDepth;

    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<100)
    {
        cout << "Wrong initialization, reseting..." << endl;
        Reset();
        return;
    }

    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
        }
    }

    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpMap->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame);

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    mState=OK;
}

void Tracking::CheckReplacedInLastFrame()       // 对上一帧中关联到keypoint 的 map point 进行检查
{
    for(int i =0; i<mLastFrame.N; i++)      // mLastFrame.N表示上一帧keypoint数目，每一个map point 都检查一次
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)     // 如果指针不为空  if(!p) = if(p==null)
        {
            MapPoint* pRep = pMP->GetReplaced();    // 如果发生替换，则会返回该替换的指针， 在Map 里不更新，只标记bad点 ？   —— 可能對應的Mappoint.cc Replace
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}


bool Tracking::TrackReferenceKeyFrame()
{
    // 计算BoW向量
    mCurrentFrame.ComputeBoW();

    // 先和Ref KeyFrame 进行BoW 匹配
    //如果有足够的matches 就进行PnP 求解
    ORBmatcher matcher(0.7,true);           //  初始化matcher：mfNNratio=0.7,   mbCheckOrientation = true.
    vector<MapPoint*> vpMapPointMatches;

    // 找到reference KeyFrame中Map point 与 current frame中 ORB 的匹配关系存储在vpMapPointMatches中，返回匹配上的总数量 ！看进去！
    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);

    if(nmatches<15)
        return false;
    // 匹配点足够多
    mCurrentFrame.mvpMapPoints = vpMapPointMatches;     // 将找到的匹配关系更新到CurrentFrame 的mvpMapPoints 中
    mCurrentFrame.SetPose(mLastFrame.mTcw); // 沿用上一帧位姿矩阵

    Optimizer::PoseOptimization(&mCurrentFrame);    //优化Current位姿mTcw，并筛选出outlier的匹配

    // 删除当前帧由Optimizer判断的outlier
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])   // 如果第i个map point 有 关联的key point
        {
            if(mCurrentFrame.mvbOutlier[i])     // 如果该点是outlier
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);  // 赋值为空表示 没有关联的key point
                mCurrentFrame.mvbOutlier[i]=false;  // 恢复flag为false
                pMP->mbTrackInView = false;     // 表示在ORBmatcher.cc的SearchByProjection中不使用该点；在track 中不使用该点
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;  // 将本帧编号作为该点最近一次被看到的帧编号
                nmatches--;     // 有效匹配点数减1
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)        // Obsercations() 返回nObs，即检查该点是否是辅助点，是辅助点就不加入Map中
                nmatchesMap++;      // 可以映射到Ｍap的点计数器加1
        }
    }

    return nmatchesMap>=10;     // 超过10个就认为Tracking 上了
}

void Tracking::UpdateLastFrame()        // 对上一帧状态进行更新,得到预测的pose?,再建立辅助点用于之后匹配,
{
    // 根据参考关键帧更新位姿
    KeyFrame* pRef = mLastFrame.mpReferenceKF;      // 上一帧的参考关键帧
    cv::Mat Tlr = mlRelativeFramePoses.back();  // 获取上一帧的Ref->camera 位姿————就是上一個Tcr

    mLastFrame.SetPose(Tlr*pRef->GetPose());    // Tlcw = Tlcr * Tlrw,上一幀中相機在世界坐標系的位姿

    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || !mbOnlyTracking)  // 上一帧新建了keyframe 或者 使用单目 或者 要求建图
        return;

    // 生成视觉里程计的MapPoints
    // 按照深度对点进行排序
    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);    // 至少提供上一帧keypoint大小的空间
    for(int i=0; i<mLastFrame.N;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));        // 距离为正，有效
        }
    }

    if(vDepthIdx.empty())   // 没有有效的点
        return;

    sort(vDepthIdx.begin(),vDepthIdx.end());

    // 插入所有close point（深度小于给定门限的点）
    // 如果close point 数量少于100, 就取最近的100个点    （越近越准确）
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;        // 第j个pair的第2个元素，即在原keypoint集合中的序号，记为i

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i];     // 关联到该keypoint[i]的map point
        if(!pMP)    // 没有mappoint 关联到该keypoints[i]
            bCreateNew = true;      // 可以为它创建新的辅助点
        else if(pMP->Observations()<1)      // 有map point点关联到,但观察到它的关键帧在eraser中被删除
        {
            bCreateNew = true;      // 可以为它创建新的辅助点
        }

        if(bCreateNew)  // 决定创建了
        {
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);    // 用上一幀映射出新的map point (用於投影到Current Frame，然後輔助匹配)
            MapPoint* pNewMP = new MapPoint(x3D,mpMap,&mLastFrame,i);

            mLastFrame.mvpMapPoints[i]=pNewMP;  // 将新生成的Mappoint 与该keypoint 关联

            mlpTemporalPoints.push_back(pNewMP); // 加入mlpTemporalPoints中
            nPoints++;      // 這些新加入用於輔助匹配的點爲初始化nObs，在插入新的KeyFrame時會刪除，可能用於減小運算量
        }
        else
        {
            nPoints++;      // 不需要生成新点，计数器直接加1
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)      // 满足条件结束
            break;
    }
}

bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9,true);       //  mfNNratio=0.9,   mbCheckOrientation = true.

    // 根据参考关键帧更新上一帧
    // 在Localization Moded则生成视觉里程计点
    UpdateLastFrame();  // 对上一帧状态进行更新,再建立辅助点用于之后匹配,

    mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);   // 使用运动模型更新Current位姿

    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));     // 将当前帧的所有的key P和 map P的关联关系都清空

    // 把前一帧的映射到3D
    int th;
    if(mSensor!=System::STEREO)
        th=15;
    else
        th=7;   // 双目门限是7——匹配所用的窗的大小
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);  // 通过搜索的 匹配点的数目

    // 如果匹配上的点少，就增大窗的大小
    if(nmatches<20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR);
    }

    if(nmatches<20)     // 还是太少就不再扩大
        return false;

    // 利用获得的所有匹配点优化当前帧的位姿
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers——同TrackReferenceKeyFrame
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;     // 表示在ORBmatcher.cc的SearchByProjection中不使用该点；在track 中不使用该点
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)    // 非oulier 且nObs>0, 故nmatchesMap較少
                nmatchesMap++;
        }
    }    

    if(mbOnlyTracking)  // 若仅进行跟踪而不建图
    {
        mbVO = nmatchesMap<10;  // 記錄MM匹配點少的情況
        return nmatches>20;     // 要求至少20点
    }

    return nmatchesMap>=10;     // 返回匹配结果,建图的话10点就行
}

bool Tracking::TrackLocalMap()
{
    // 已知相机位姿的估计和当前帧中跟踪的map point
    // 对local map 进行恢复并找到与local map中的点的匹配关系 （try to find matches to points in the local map.）

    UpdateLocalMap();       // 获得文中所述 K1, K2集合 以及 Ref KeyFrame； 获得用于对 CurrentFram进行 Track的 Mappoint集合

    SearchLocalPoints();    // 获得更多与 CurrentFrame的 Key point匹配的 Map points

    // Optimize Pose
    Optimizer::PoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0;   // 前匹配上的点数

    // 更新 MapPoints Statistics
    for(int i=0; i<mCurrentFrame.N; i++)    // 对当前帧所有Key Point
    {
        if(mCurrentFrame.mvpMapPoints[i])   // 如果当前第i个Key point 有匹配的Map Point
        {
            if(!mCurrentFrame.mvbOutlier[i])    // Optimizer會輸出該Map Point不是outlier
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();     // 跟踪计数器mnFound + 1     ？
                if(!mbOnlyTracking)     // 开启了建图功能
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)     // 有其他幀觀察到該Map point
                        mnMatchesInliers++;     // 当前匹配上的点数+1
                }
                else
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO)        // 不知道
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);   // 置NULL, 但是沒有恢復mvbOutlier 不知道

        }
    }

    // 判决Tracking是否成功
    // 按文中mMaxFrames为20,
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)      // 如果最近发生relocalization，要求多于50点
        return false;

    if(mnMatchesInliers<30)     // 稳定跟踪下，要求多于30点
        return false;
    else
        return true;
}


bool Tracking::NeedNewKeyFrame()
{
    if(mbOnlyTracking)      // 没有开建图，就不插入
        return false;

    // 如果 Local Mapping被 Loop Closure冻结 不插关键帧
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    const int nKFs = mpMap->KeyFramesInMap();       // 返回 mspKeyFrames.size()

    // 如果距离最近一次 relocalization还没有过去足够的帧数就不插入 keyframes
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)     // mMaxFrames就是相机帧率; 後一條表示前期有機會就插入keyframe，後期主要依賴前一條件
        return false;

    // 在 reference keyframe跟踪Map point
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);     // 得到Ref Keyframe中至少能被 nMinObs个关键帧看到的 Map points个数

    // Local Mapping 是否接受 keyframes？——查看Local Mapper 是否空闲
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // 检查 "close" points被 track到和没被 track到的数量 how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;
    if(mSensor!=System::MONOCULAR)      // 如果不是单目
    {
        for(int i =0; i<mCurrentFrame.N; i++)       // 对每一个关键帧的Key poinit
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)     // 如果是“close” 点
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])       // 该点被关联到 Map point且不是异常的关联
                    nTrackedClose++;        // 跟踪点计数器+1
                else
                    nNonTrackedClose++; // 未跟踪点计数器+1
            }
        }
    }

    bool bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);     // 跟上的点少于100个 且未跟上的大于70个

    // Thresholds
    float thRefRatio = 0.75f;   // 强制为float ———— mnFound/mnVisible 的門限
    if(nKFs<2)      // 如果地图的关键帧少于2个
        thRefRatio = 0.4f;

    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion；文中条件1的意思
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;

    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle； 文中条件2的意思
    const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);        // mMinFrames = 0，意为只要idle就可以插

    //Condition 1c: tracking is weak； 非单目且跟踪效果很堪忧——（Inliers少于Ref KeyFrame的1/4，跟上的点少于100个 且未跟上的大于70个）
    const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;

    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    // 前半部分和上一个条件相似，是文中条件4的意思（改成了75%）加上或者跟踪效果堪忧，同时要求当前的匹配大于15点（文章中50点）
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio|| bNeedToInsertClose) && mnMatchesInliers>15);

    if((c1a||c1b||c1c)&&c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA  与文中叙述一致
        if(bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR)
            {
                if(mpLocalMapper->KeyframesInQueue()<3)     // 返回mlNewKeyFrames.size()  已插入的关键帧的数量，如果少于3则赶紧插入
                    return true;
                else
                    return false;
            }
            else
                return false;       // 是单目就不插入关键帧
        }
    }
    else
        return false;
}

void Tracking::CreateNewKeyFrame()
{
    if(!mpLocalMapper->SetNotStop(true))    // 设置mbNotStop = true 失败(因为mbStopped = true 表示已经停止了)
        return;

    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB); // 生成新的KeyFrame

    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    if(mSensor!=System::MONOCULAR)
    {
        mCurrentFrame.UpdatePoseMatrices(); // 更新当前帧位姿矩阵

        // 按点的深度对点进行排序.
        // 将那些深度小于 mThDepth的Mappoint 加入.
        // close点少于100个则取最近的100个点
        vector<pair<float,int> > vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);     // 开辟与 CurrentFrame的 Keypoint数量相同的空间
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)     // 深度有效
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())      // 只要不是没有有效深度
        {
            sort(vDepthIdx.begin(),vDepthIdx.end());    // 按深度排序

            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];  // 对应第i个Keypoint的Mappoint
                if(!pMP)        // 该Key point没有对应的Map point
                    bCreateNew = true;      // 那就创建一个
                else if(pMP->Observations()<1)      // 或者该Key point被看到的帧数为0
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);       // 将Mappoint 置NULL
                }

                if(bCreateNew)      // 决定要新建点
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);     // 將 Keypoint映射到3D空間中
                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpMap);     // 生成並初始化Mappoint的實例，包含了該點所屬的關鍵幀，被看到的幀數nObs等許多屬性
                    pNewMP->AddObservation(pKF,i);      // 建立新建Mappoint與pKF 和第i 個Keypoint的對應關系
                    pKF->AddMapPoint(pNewMP,i);         // 建立KeyFrame 與新建Mappoint 和 Keypoint 的對應關系 —— mvpMapPoints[i] = pNewMP, 觀察到第i個Keypoint的Mappoint是pNewMP
                    pNewMP->ComputeDistinctiveDescriptors();    // 找到該點和其他描述子的最小中值距離————用作Mappoint 的描述姿子
                    pNewMP->UpdateNormalAndDepth();     // 更新法向和深度
                    mpMap->AddMapPoint(pNewMP);         // 添加pNewMP 到本map 的mspMapPoints 中

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;   // 當前幀第i個Mappoint 是pNewMP
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                if(vDepthIdx[j].first>mThDepth && nPoints>100)  // 若該點的深度大門限且已經對超過100個點進行判決，則停止
                    break;
            }
        }
    }

    mpLocalMapper->InsertKeyFrame(pKF);     // 將pKF插入KeyFrame

    mpLocalMapper->SetNotStop(false);       // 表示建圖未完成

    mnLastKeyFrameId = mCurrentFrame.mnId;  // 將CurrentFrame 置爲最近一幀關鍵幀，id也同步
    mpLastKeyFrame = pKF;
}

void Tracking::SearchLocalPoints()
{
    // 标记一下哪些是已经匹配上的Map point
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)     // 若有匹配上key point的 map point
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);        // 坏点置NULL
            }
            else
            {
                pMP->IncreaseVisible();     // 跟踪计数器mnVisible+1
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;      // 标记该map point最近一次是被CurrentFrame看到
                pMP->mbTrackInView = false;     // 表示在ORBmatcher.cc的SearchByProjection中不使用该点；在track 中不使用该点
            }
        }
    }

    int nToMatch=0;

    // 将所有Map poing投影到frame中，检查可视性
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)      //对于已经匹配上的map point不进行搜索
            continue;
        if(pMP->isBad())        // 对于坏点不进行搜索
            continue;
        // 映射(this fills MapPoint variables for matching)
        if(mCurrentFrame.isInFrustum(pMP,0.5))  // 根据文中 V节 D小节 的1~5点进行筛选，返回是否可视 （边界，视角，距离，尺度）
        {
            pMP->IncreaseVisible();     // 可视则mnVisible+1   ？ 有什麼用
            nToMatch++;     //记录新增的匹配的Map point数量
        }
    }

    if(nToMatch>0)      // 如果存在新增Map points
    {
        ORBmatcher matcher(0.8);    // 设置ORBmatcher
        int th = 1;     // 搜索门限
        if(mSensor==System::RGBD)
            th=3;
        // 如果刚刚完成重定位,则进行更粗颗粒度的搜索
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)     // 帧编号相差不超过2则认为刚完成
            th=5;
        matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);     // 开始映射搜索
    }
}

void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();     // 获得文中所述 K1, K2集合 以及 Ref KeyFrame
    UpdateLocalPoints();        // 获得用于对 CurrentFram进行 Track的 Mappoint集合
}

void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();      // 清空本地MapPoint向量
    // 对当前本地关键帧逐一操作
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches(); // 得到pKF代表的帧中与 Keypoint关联的 Map points
        // 对每一个与 Keypoint关联的 Map point
        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)    // 如果pMP为空——vpMPs为空
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)      // map point已被标记为用于CurrentFrame的Track
                continue;
            if(!pMP->isBad())   // 不与当前帧的keypoint 关联，且不是坏点
            {
                mvpLocalMapPoints.push_back(pMP);   // 将其存在mvpLocalMapPoints中
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;   // 并将其标记为用于CurrentFrame的Track
            }
        }
    }
}


void Tracking::UpdateLocalKeyFrames()
{
    // 每一个 map point 给看到该点的Key Frame投票
    map<KeyFrame*,int> keyframeCounter;     // map是一个容器，提供一对一的关系
    for(int i=0; i<mCurrentFrame.N; i++)    // 對当前帧每一个Keypoint
    {
        if(mCurrentFrame.mvpMapPoints[i])       // MapPoint associated to ith keypoint 的关联存在
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP->isBad())       // 不是坏点
            {
                const map<KeyFrame*,size_t> observations = pMP->GetObservations();      // 返回的是观察到该 Mappoint 的所有KeyFrames以及（该mappoint在keyframe中的指针还是就是指针？）
                for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)  //const_iterator迭代器。这个迭代器是可以自己增加的，但是其所指向的元素是不可以被改变的，迭代器可以不是数
                    keyframeCounter[it->first]++;       // it->first 是KeyFrame， 指定KeyFrame的map<KeyFrame*,int> ，得到的就是int
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i]=NULL;     // 没有关联，指针置NULL
            }
        }
    }

    if(keyframeCounter.empty())     // 所有帧都找不到对应关系
        return;

    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);     // static_cast显式完成类型转换, 将NULL转换为KeyFrame* 型

    mvpLocalKeyFrames.clear();      // 清空mvpLocalKeyFrames
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // 所有观察到一个map point的关键帧都包含到local map中， 并且检测共享最多点的那一个关键帧
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())    // 坏帧就不取
            continue;

        if(it->second>max)      // 找到与当前帧share最多map point的关键帧 之后将其设为文中提到的Ref KeyFrame
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);     // 存入本地关键帧
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;         // 标记该关键帧是当前帧在Track中的相关帧 也即文中K1 KeyFrame 集合
    }


    // 包含进一些未被包含的Keyframe，它们是已经被包含的那些keyframs的近邻 ———— 同时考虑了外观和时间远近
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80)
            break;

        KeyFrame* pKF = *itKF;

        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);        //返回该本地关键帧的前N个最佳共视关键帧 向量， 也即文中K2 KeyFrame集合

        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())      // 不是坏帧
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)      // 如果该共视关键帧不是当前帧的相关帧，即不在K1中的 K2中的关键帧
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);      // 存为本地关键帧
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;  // 设为当前帧的相关帧
                    break;      // 只存最好的那一个本地关键帧的共视关键帧
                }
            }
        }

        const set<KeyFrame*> spChilds = pKF->GetChilds();   // 返回该本地关键帧的生成树子节点
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)      // 与上面类似，将良好的生成树的子节点存起来
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        KeyFrame* pParent = pKF->GetParent();       // 同理，将良好的亲辈节点存起来
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }

    }

    if(pKFmax)      // 只要K1集合不是空集——否则最大共享map point数量才有可能为0，导致pKFmax为空
    {
        mpReferenceKF = pKFmax;     // 将其设为文中提到的Ref KeyFrame
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

bool Tracking::Relocalization()
{
    // 计算当前帧的BoW向量
    mCurrentFrame.ComputeBoW();

    // 跟踪Lost时进行重定位
    // 搜索keyframe集合找用于重定位的candidates——与CurrentFrame相似的
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

    if(vpCandidateKFs.empty())  // 返回为空说明没有找到candidate
        return false;

    const int nKFs = vpCandidateKFs.size();

    // 对每一个cadidate都进行ORB 匹配
    // 如果匹配点足够多就进行 PnP 求解
    ORBmatcher matcher(0.75,true);      // ORB matcher的设置

    vector<PnPsolver*> vpPnPsolvers;    // 生成PnP求解器
    vpPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;  // 存储match上的点
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;   // 記錄KeyFrame是否有效
    vbDiscarded.resize(nKFs);

    int nCandidates=0;  // 记录keyframe中的Candidate数量
    // 一个一个试
    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())        // keyframe 是壞幀 在KeyFrameCull 和 Eraser中有設置 ？
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);    // 通過SearchByBoW，找到第i個CandidateKF 中Mapoint 與CurrentFrame 中Keypoint 間的關聯存到vvpMapPointMatches
            if(nmatches<15)
            {
                vbDiscarded[i] = true;      // match点少于15则无效
                continue;
            }
            else
            {
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);    // match点大于15则进行PnP求解
                pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);  // 设置每一個點數足夠多的匹配的求解器
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // 不断进行P4P RANSAC 迭代        ？
    // 直到有足够多的inlier支持该相机位姿
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    while(nCandidates>0 && !bMatch)     // 只要还有candidate且支持点数量不达标（50以上）就迭代
    {
        for(int i=0; i<nKFs; i++)       // 还是逐帧來，因爲循環中索引順序和幀順序是一樣的，不是全都做一遍//
        {
            if(vbDiscarded[i])
                continue;

            // 對mCurrentFrame 和vvpMapPointMatches[i] 进行5次RANSAC迭代
            vector<bool> vbInliers;     // 迭代得到的vvpMapPointMatches[i] 中inlier的記錄
            int nInliers;
            bool bNoMore;

            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);   // 得到相机位姿，迭代点数不够或者迭代次数大于最大值时bNoMore为True; vbInliers是bool vecor表明哪些是inlier

            // 到了最大值都找不到 忽略該 keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // 获得了相机位姿, 优化
            if(!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);     // 复制成爲currentframe的位姿

                set<MapPoint*> sFound;  // 用於區分由BoW 和Projection 得到的Mappoint

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];     // 將vvpMapPointMatches[i]中第j個匹配 作爲CurrentFrame 的Mappoint-Keypoint匹配
                        sFound.insert(vvpMapPointMatches[i][j]);    // 插入sFound中
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);    // 对当前帧进行位姿优化 返回inlier的correspondence数， 會指出哪些是outlier點

                if(nGood<10)    // 如果得到的correspondence很少，就結束這一波
                    continue;

                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])    // 沒有恢復標志爲False
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // 如果基於SearchByBoW位姿優化得到的inliers 仍不夠多, SearchByProjection再一次映射以后在更大的窗中搜索并优化
                if(nGood<50)
                {
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);    // 反投影增加mappoint

                    if(nadditional+nGood>=50)   // 兩種方法總點數足夠
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // nGood还不够，则使用SearchByProjection在窄一些的窗中搜索————剛才超過50說明是可以有的
                        // 相机已经被优化了
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear(); // Optimizer會更新outlier， 這裏更新一下
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);  //
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // 如果点数足够，最后进行一次优化
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }


                // 如果位姿被足够多 inliers支持则停止Ransac，继续下一拨
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)     // 如果没有一个位姿估计被通过就发返回false
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }

}

void Tracking::Reset()
{

    cout << "System Reseting" << endl;
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    // Reset Loop Closing
    cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    // 清空地图（所有地图点和关键帧）
    mpMap->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if(mpInitializer)   // 地图点初始化
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    mlRelativeFramePoses.clear();   //相对位姿
    mlpReferences.clear();  //清空参考关键帧
    mlFrameTimes.clear();   //清除每帧时间戳
    mlbLost.clear(); // 清除跟踪失败的flag

    if(mpViewer)
        mpViewer->Release();    // viewer相关
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}



} //namespace ORB_SLAM
