/**
 * @file OFIQImpl.cpp
 *
 * @copyright Copyright (c) 2024  Federal Office for Information Security, Germany
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * @author OFIQ development team
 */

#include "Configuration.h"
#include "Executor.h"
#include "ofiq_lib_impl.h"
#include "OFIQError.h"
#include "FaceMeasures.h"
#include "utils.h"
#include "image_io.h"
#include <chrono>
using hrclock = std::chrono::high_resolution_clock;

using namespace std;
using namespace OFIQ;
using namespace OFIQ_LIB;
using namespace OFIQ_LIB::modules::measures;


OFIQImpl::OFIQImpl():m_emptySession({this->dummyImage, this->dummyAssement}) {}

ReturnStatus OFIQImpl::initialize(const std::string& configDir, const std::string& configFilename)
{
    try
    {
        this->config = std::make_unique<Configuration>(configDir, configFilename);
        CreateNetworks();
        m_executorPtr = CreateExecutor();
    }
    catch (const OFIQError & ex)
    {
        return {ex.whatCode(), ex.what()};
    }
    catch (const std::exception & ex)
    {
        return {ReturnCode::UnknownError, ex.what()};
    }

    return ReturnStatus(ReturnCode::Success);
}

ReturnStatus OFIQImpl::scalarQuality(const OFIQ::Image& face, double& quality)
{
    FaceImageQualityAssessment assessments;

    if (auto result = vectorQuality(face, assessments); 
        result.code != ReturnCode::Success)
        return result;

    if(assessments.qAssessments.find(QualityMeasure::UnifiedQualityScore) != assessments.qAssessments.end())
        quality = assessments.qAssessments[QualityMeasure::UnifiedQualityScore].scalar;
    else
    {
        // just as an example - the scalarQuality is an average of all valid scalar measurements
        double sumScalars = 0;
        int numScalars = 0;
        for (auto const& [measureName, aResult] : assessments.qAssessments)
        {
            if (aResult.scalar != -1)
            {
                sumScalars += aResult.scalar;
                numScalars++;
            }
        }
        quality = numScalars != 0 ? sumScalars / numScalars : 0;
    }

    return ReturnStatus(ReturnCode::Success);
}

void OFIQImpl::performPreprocessing(Session& session)
{
    std::chrono::time_point<hrclock> tic;

    log("\t1. detectFaces ");
    tic = hrclock::now();

    std::vector<OFIQ::BoundingBox> faces = networks->faceDetector->detectFaces(session);
    if (faces.empty())
    {
        log("\n\tNo faces were detected, abort preprocessing\n");
        throw OFIQError(ReturnCode::FaceDetectionError, "No faces were detected");
    }
    log(std::to_string(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            hrclock::now() - tic).count()) + std::string(" ms "));

    session.setDetectedFaces(faces);
    log("2. estimatePose ");
    tic = hrclock::now();

    session.setPose(networks->poseEstimator->estimatePose(session));

    log(std::to_string(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            hrclock::now() - tic).count()) + std::string(" ms "));

    log("3. extractLandmarks ");
    tic = hrclock::now();

    session.setLandmarks(networks->landmarkExtractor->extractLandmarks(session));

    log(std::to_string(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            hrclock::now() - tic).count()) + std::string(" ms "));

    log("4. alignFaceImage ");
    tic = hrclock::now();
    // aligned face requires the landmarks of the face thus it must come after the landmark extraction.
    alignFaceImage(session);
    log(std::to_string(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            hrclock::now() - tic).count()) + std::string(" ms "));

    log("5. getSegmentationMask ");
    tic = hrclock::now();
    // segmentation results for face_parsing
    session.setFaceParsingImage(OFIQ_LIB::copyToCvImage(
        networks->segmentationExtractor->GetMask(
            session,
            OFIQ_LIB::modules::segmentations::SegmentClassLabels::face),
        true));
    log(std::to_string(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            hrclock::now() - tic).count()) + std::string(" ms "));

    log("6. getFaceOcclusionMask ");
    tic = hrclock::now();
    session.setFaceOcclusionSegmentationImage(OFIQ_LIB::copyToCvImage(
        networks->faceOcclusionExtractor->GetMask(
            session,
            OFIQ_LIB::modules::segmentations::SegmentClassLabels::face),
        true));
    log(std::to_string(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            hrclock::now() - tic).count()) + std::string(" ms "));

    static const std::string alphaParamPath = "params.measures.FaceRegion.alpha";
    double alpha = 0.0f;
    try
    {
        alpha = this->config->GetNumber(alphaParamPath);
    }
    catch(OFIQError&)
    {
        alpha = 0.0f;
    }

    log("7. getAlignedFaceMask ");
    tic = hrclock::now();

    session.setAlignedFaceLandmarkedRegion(
         OFIQ_LIB::modules::landmarks::FaceMeasures::GetFaceMask(
            session.getAlignedFaceLandmarks(),
            session.getAlignedFace().rows,
            session.getAlignedFace().cols,
            (float)alpha
         )
    );
    log(std::to_string(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            hrclock::now() - tic).count()) + std::string(" ms "));

    log("\npreprocessing finished\n");
}

void OFIQImpl::alignFaceImage(Session& session) const
{
    auto landmarks = session.getLandmarks();
    OFIQ::FaceLandmarks alignedFaceLandmarks;
    alignedFaceLandmarks.type = landmarks.type;
    cv::Mat transformationMatrix;
    cv::Mat alignedBGRimage = alignImage(session.image(), landmarks, alignedFaceLandmarks, transformationMatrix);

    session.setAlignedFace(alignedBGRimage);
    session.setAlignedFaceLandmarks(alignedFaceLandmarks);
    session.setAlignedFaceTransformationMatrix(transformationMatrix);

}

ReturnStatus OFIQImpl::vectorQuality(
    const OFIQ::Image& image, OFIQ::FaceImageQualityAssessment& assessments)
{
    auto session = Session(image, assessments);
    return vectorQualityViaSession(session);
}

ReturnStatus OFIQImpl::vectorQuality(
    const OFIQ::Image& image, OFIQ::FaceImageQualityAssessment& assessments, OFIQ::ExposedSession& exposedSession)
{
    auto session = new Session(image, assessments);
    exposedSession.session = (void*)session;
    return vectorQualityViaSession(*session);
}

OFIQ_EXPORT ExposedSession::~ExposedSession() {
    delete (Session*)session;
}
OFIQ_EXPORT std::vector<BoundingBox> ExposedSession::getDetectedFaces() {
    return ((Session*)session)->getDetectedFaces();
}
OFIQ_EXPORT std::array<double, 3> ExposedSession::getPose() {
    return ((Session*)session)->getPose();
}
OFIQ_EXPORT FaceLandmarks ExposedSession::getLandmarks() {
    return ((Session*)session)->getLandmarks();
}
OFIQ_EXPORT FaceLandmarks ExposedSession::getAlignedFaceLandmarks() {
    return ((Session*)session)->getAlignedFaceLandmarks();
}
OFIQ_EXPORT cv::Mat ExposedSession::getAlignedFaceTransformationMatrix() {
    return ((Session*)session)->getAlignedFaceTransformationMatrix();
}
OFIQ_EXPORT cv::Mat ExposedSession::getAlignedFace() {
    return ((Session*)session)->getAlignedFace();
}
OFIQ_EXPORT cv::Mat ExposedSession::getAlignedFaceLandmarkedRegion() {
    return ((Session*)session)->getAlignedFaceLandmarkedRegion();
}
OFIQ_EXPORT cv::Mat ExposedSession::getFaceParsingImage() {
    return ((Session*)session)->getFaceParsingImage();
}
OFIQ_EXPORT cv::Mat ExposedSession::getFaceOcclusionSegmentationImage() {
    return ((Session*)session)->getFaceOcclusionSegmentationImage();
}

ReturnStatus OFIQImpl::vectorQualityViaSession(
    Session& session)
{
    try
    {
        log("perform preprocessing:\n");
        performPreprocessing(session);
    }
    catch (const OFIQError& e)
    {
        log("OFIQError: " + std::string(e.what()) + "\n");
        for (const auto& measure : m_executorPtr->GetMeasures() )
        {
            auto qualityMeasure = measure->GetQualityMeasure();
            switch (qualityMeasure)
            {
            case QualityMeasure::Luminance:
                session.assessment().qAssessments[QualityMeasure::LuminanceMean] =
                { 0, -1, OFIQ::QualityMeasureReturnCode::FailureToAssess };
                session.assessment().qAssessments[QualityMeasure::LuminanceVariance] =
                { 0, -1, OFIQ::QualityMeasureReturnCode::FailureToAssess };
                break;
            case QualityMeasure::CropOfTheFaceImage:
                session.assessment().qAssessments[QualityMeasure::LeftwardCropOfTheFaceImage] =
                { 0, -1, OFIQ::QualityMeasureReturnCode::FailureToAssess };
                session.assessment().qAssessments[QualityMeasure::RightwardCropOfTheFaceImage] =
                { 0, -1, OFIQ::QualityMeasureReturnCode::FailureToAssess };
                session.assessment().qAssessments[QualityMeasure::MarginBelowOfTheFaceImage] =
                { 0, -1, OFIQ::QualityMeasureReturnCode::FailureToAssess };
                session.assessment().qAssessments[QualityMeasure::MarginAboveOfTheFaceImage] =
                { 0, -1, OFIQ::QualityMeasureReturnCode::FailureToAssess };
                break;
            case QualityMeasure::HeadPose:
                session.assessment().qAssessments[QualityMeasure::HeadPoseYaw] =
                { 0, -1, OFIQ::QualityMeasureReturnCode::FailureToAssess };
                session.assessment().qAssessments[QualityMeasure::HeadPosePitch] =
                { 0, -1, OFIQ::QualityMeasureReturnCode::FailureToAssess };
                session.assessment().qAssessments[QualityMeasure::HeadPoseRoll] =
                { 0, -1, OFIQ::QualityMeasureReturnCode::FailureToAssess };
                break;
            default:
                session.assessment().qAssessments[measure->GetQualityMeasure()] =
                { 0, -1, OFIQ::QualityMeasureReturnCode::FailureToAssess };
                break;
            }
        }

        return { e.whatCode(), e.what() };
    }

    log("execute assessments:\n");
    m_executorPtr->ExecuteAll(session);

    return ReturnStatus(ReturnCode::Success);
}

OFIQ_EXPORT std::shared_ptr<Interface> Interface::getImplementation()
{
    return std::make_shared<OFIQImpl>();
}

OFIQ_EXPORT void Interface::getVersion(int& major, int& minor, int& patch) const
{
    major = int(OFIQ_VERSION_MAJOR);
    minor = int(OFIQ_VERSION_MINOR);
    patch = int(OFIQ_VERSION_PATCH);
    return;
}
