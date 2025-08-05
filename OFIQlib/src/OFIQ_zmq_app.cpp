/**
 * @file OFIQ_zmq_app.cpp
 *
 * @copyright Copyright (c) 2024 da/sec
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
 * @author Torsten Schlett
 *
 * @note ZmqFork-NOTE: This is an addition of the ZeroMQ/Python fork.
 */

#if defined _WIN32 && defined OFIQ_EXPORTS
#undef OFIQ_EXPORTS
#endif

// OFIQ includes:
#include "ofiq_lib.h"
#include "image_io.h"
#include "utils.h"

// Standard includes:
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <bit>

// External includes:
#include <opencv2/opencv.hpp>
#include <zmq.h>

constexpr int SUCCESS = 0;
constexpr int FAILURE = 1;
constexpr uint64_t expected_message_format_version = 1;

using namespace std;
using namespace OFIQ;
using namespace OFIQ_LIB;

template<typename T>
T network_byteswap(T value) {
    if constexpr ((std::endian::native == std::endian::big) || (sizeof(value) == 1))
        return value;
    else {
        // Note: "return std::byteswap(value);" does not appear to be readily available.
        if constexpr (sizeof(T) == 2) {
            uint16_t tmp = *(uint16_t*)&value;
            tmp = (tmp >> 8) | (tmp << 8);
            return *(T*)&tmp;
        } else if constexpr (sizeof(T) == 4) {
            uint32_t tmp = *(uint32_t*)&value;
            tmp = ( tmp << (8 * 3)) | // 0xFF000000
                  ((tmp << (8 * 1)) &    0x00FF0000) |
                  ((tmp >> (8 * 1)) &    0x0000FF00) |
                  ( tmp >> (8 * 3));  // 0x000000FF
            return *(T*)&tmp;
        } else if constexpr (sizeof(T) == 8) {
            uint64_t tmp = *(uint64_t*)&value;
            tmp = ( tmp << (8 * 7)) | // 0xFF00000000000000
                  ((tmp << (8 * 5)) &    0x00FF000000000000) |
                  ((tmp << (8 * 3)) &    0x0000FF0000000000) |
                  ((tmp << (8 * 1)) &    0x000000FF00000000) |
                  ((tmp >> (8 * 1)) &    0x00000000FF000000) |
                  ((tmp >> (8 * 3)) &    0x0000000000FF0000) |
                  ((tmp >> (8 * 5)) &    0x000000000000FF00) |
                  ( tmp >> (8 * 7)); //  0x00000000000000FF
            return *(T*)&tmp;
        }
    }
}

struct Reader {
    uint8_t* full_data = nullptr;
    uint8_t* cursor_data = nullptr;
    size_t full_size = 0;
    size_t cursor_index = 0;
    size_t cursor_size = 0;

    Reader(std::vector<uint8_t>& full_message_data) {
        full_data = &full_message_data[0];
        cursor_data = full_data;
        full_size = full_message_data.size();
        cursor_index = 0;
        cursor_size = 0;
    }

    bool next(size_t next_size) {
        cursor_data += cursor_size;
        cursor_index += cursor_size;
        cursor_size = next_size;
        return (cursor_index + next_size) <= full_size;
    }

    bool read(size_t next_size, void* copy_to) {
        bool size_ok = next(next_size);
        if (size_ok)
            std::memcpy(copy_to, cursor_data, next_size);
        return size_ok;
    }

    template<typename T>
    bool read_scalar(T& copy_to) {
        bool size_ok = read(sizeof(copy_to), &copy_to);
        if (size_ok)
            copy_to = network_byteswap(copy_to);
        return size_ok;
    }

    void check_end(uint8_t command_type) {
        if (!next(0)) {
            size_t leftover_size = full_size - cursor_index;
            cerr << "[OFIQ_zmq_app][WARNING] Reader::check_end - Unexpected leftover data at the end of the message: " << leftover_size << " bytes (command type " << (uint32_t)command_type << ")" << endl;
        }
    }
};

struct Writer {
    std::vector<uint8_t> full_data;

    void write(void* data, size_t size) {
        size_t prior_size = full_data.size();
        full_data.resize(prior_size + size);
        std::memcpy(&full_data[prior_size], data, size);
    }

    template<typename T>
    void write_scalar(T value) {
        value = network_byteswap(value);
        write(&value, sizeof(value));
    }

    void write_header(uint8_t command_type) {
        write_scalar(expected_message_format_version);
        write_scalar(command_type);
    }

    void write_ofiq_bounding_box(BoundingBox& boundingBox) {
        write_scalar((int16_t)boundingBox.xleft);
        write_scalar((int16_t)boundingBox.ytop);
        write_scalar((int16_t)boundingBox.width);
        write_scalar((int16_t)boundingBox.height);
        write_scalar((uint8_t)boundingBox.faceDetector);
    }

    void write_ofiq_landmarks(FaceLandmarks& faceLandmarks) {
        write_scalar((uint8_t)faceLandmarks.type);
        {
            Landmarks& landmarks = faceLandmarks.landmarks;
            write_scalar((uint32_t)landmarks.size());
            for (LandmarkPoint& landmarkPoint : landmarks) {
                write_scalar((int16_t)landmarkPoint.x);
                write_scalar((int16_t)landmarkPoint.y);
            }
        }
    }

    template<typename T>
    void _write_cv_mat__data(cv::Mat& mat) {
        int count = mat.rows * mat.cols * mat.channels();
        T* value_ptr = (T*)mat.ptr();
        for (int i = 0; i < count; ++i)
            write_scalar(value_ptr[i]);
    }

    void _write_cv_mat__continuous(cv::Mat& mat) {
        assert(mat.isContinuous());
        bool supported = true;
        int channels = mat.channels(); // Cf. // uint8_t channels = 1 + (type >> CV_CN_SHIFT);
        int depth = mat.depth(); // Cf. // uint8_t depth = type & CV_MAT_DEPTH_MASK;
        uint8_t depth_write = 255;
        switch (depth) { // (depth_write should happen to be identical to depth for the valid cases, but that is unimportant.)
            case CV_8U:  depth_write = 0; break;
            case CV_8S:  depth_write = 1; break;
            case CV_16U: depth_write = 2; break;
            case CV_16S: depth_write = 3; break;
            case CV_32S: depth_write = 4; break;
            case CV_32F: depth_write = 5; break;
            case CV_64F: depth_write = 6; break;
            // Unsupported: // case CV_16F: depth_write = 7; break;
            default:
                depth_write = 255;
                supported = false;
                cerr << "[OFIQ_zmq_app][ERROR] Writer::write_cv_mat: Unsupported depth value. depth=" << (uint32_t)depth << endl;
            break;
        }
        if (supported) {
            write_scalar((int32_t)mat.cols);
            write_scalar((int32_t)mat.rows);
            write_scalar((int32_t)channels);
            write_scalar((uint8_t)depth_write);
            int size = mat.cols * mat.rows * (int)channels;
            switch (depth) {
                case CV_8U: _write_cv_mat__data<uint8_t>(mat); break;
                case CV_8S: _write_cv_mat__data<int8_t>(mat); break;
                case CV_16U: _write_cv_mat__data<uint16_t>(mat); break;
                case CV_16S: _write_cv_mat__data<int16_t>(mat); break;
                case CV_32S: _write_cv_mat__data<int32_t>(mat); break;
                case CV_32F: _write_cv_mat__data<float>(mat); break;
                case CV_64F: _write_cv_mat__data<double>(mat); break;
                default:
                    cerr << "[OFIQ_zmq_app][ERROR] Writer::write_cv_mat: Code should not be reachable." << endl;
                break; // Should not be reachable.
            }
        } else {
            write_scalar((int32_t)-1);  // Unsupported-cv::Mat marker.
        }
    }

    void write_cv_mat(cv::Mat& mat) {
        if (mat.isContinuous()) {
            _write_cv_mat__continuous(mat);
        } else {
            cv::Mat continuous_mat = mat.clone();
            _write_cv_mat__continuous(continuous_mat);
        }
    }
};

// Cf. https://libzmq.readthedocs.io/en/latest/zmq_msg_more.html
int zmq_receive_multipart_message(void *socket, std::vector<uint8_t>& full_message_data) {
    full_message_data.resize(0);
    zmq_msg_t part;
    {
        int timeout_ms = 60000;
        zmq_setsockopt(socket, ZMQ_RCVTIMEO, &timeout_ms, sizeof(timeout_ms));
    }
    while (true) {
        //  Create an empty 0MQ message to hold the message part
        zmq_msg_init(&part);
        //  Block until a message is available to be received from socket
        int rc = zmq_msg_recv(&part, socket, 0);
        if (rc == -1) {
            if (errno == EAGAIN) {
                return 0;
            } else {
                cerr << "[OFIQ_zmq_app][ERROR] zmq_msg_recv: " << zmq_strerror(errno) << endl;
                return -1;
            }
        }

        size_t part_size = zmq_msg_size(&part);
        size_t prior_full_size = full_message_data.size();
        full_message_data.resize(prior_full_size + part_size);
        void* part_data = zmq_msg_data(&part);
        std::memcpy(&full_message_data[prior_full_size], part_data, part_size);

        int more = zmq_msg_more(&part);
        zmq_msg_close(&part);
        if (more == 0)
            return 1; // Message end
    }
}

void zmq_send_message(void *socket, Writer& writer) {
    zmq_msg_t msg;
    {
        int rc = zmq_msg_init_size(&msg, writer.full_data.size());
        if (rc == -1) {
            cerr << "[OFIQ_zmq_app][ERROR] zmq_send_message - zmq_msg_init_size: " << zmq_strerror(errno) << endl;
        }
        std:memcpy(zmq_msg_data(&msg), &writer.full_data[0], writer.full_data.size());
    }
    {
        int rc = zmq_msg_send(&msg, socket, 0);
        if (rc == -1) {
            cerr << "[OFIQ_zmq_app][ERROR] zmq_send_message - zmq_msg_send: " << zmq_strerror(errno) << endl;
            zmq_msg_close(&msg);
        }
    }
}

int process_message(void *socket, std::shared_ptr<Interface>& implPtr, std::vector<uint8_t>& full_message_data) {
    Reader reader(full_message_data);
    uint64_t message_format_version = 0;
    if (!reader.read_scalar(message_format_version)) return -1;
    if (message_format_version != expected_message_format_version) return -1;
    uint8_t command_type = 0;
    if (!reader.read_scalar(command_type)) return -1;
    switch (command_type) {
        case 0: { // Command type 0: Ping (does nothing)
            reader.check_end(command_type);
            // Send the confirmation:
            {
                Writer writer;
                writer.write_header(command_type);
                zmq_send_message(socket, writer);
            }
        } break;
        case 1: { // Command type 1: Shutdown.
            reader.check_end(command_type);
            cout << "[OFIQ_zmq_app][INFO] Received shutdown command." << endl;
            // Send the confirmation:
            {
                Writer writer;
                writer.write_header(command_type);
                zmq_send_message(socket, writer);
            }
        } return 1;
        case 2: { // Command type 2: Image processing
            // Unpack RGB image data:
            uint32_t message_image_id = 0;
            if (!reader.read_scalar(message_image_id)) return -1;
            uint16_t width = 0;
            uint16_t height = 0;
            if (!reader.read_scalar(width)) return -1;
            if (!reader.read_scalar(height)) return -1;
            Image image;
            image.width = width;
            image.height = height;
            image.depth = 24;
            {
                uint32_t size = width;
                size *= height;
                size *= 3;
                image.data = std::shared_ptr<uint8_t[]>(new uint8_t[size]);
                if (!reader.read(size, image.data.get())) return -1;
            }
            reader.check_end(command_type);
            // Process the image:
            FaceImageQualityAssessment assessment;
            ExposedSessionZmqFork session;
            ReturnStatus retStatus = implPtr->vectorQualityZmqFork(image, assessment, session);
            // Serialize result message data:
            Writer writer;
            {
                // Result data header:
                writer.write_header(command_type);
                writer.write_scalar(message_image_id);
                uint8_t processing_success = retStatus.code == ReturnCode::Success ? 1 : 0;
                writer.write_scalar(processing_success);
                if (processing_success) {
                    // Result data part 1: OFIQ::FaceImageQualityAssessment::boundingBox (OFIQ::BoundingBox)
                    {
                        BoundingBox& boundingBox = assessment.boundingBox;
                        writer.write_ofiq_bounding_box(boundingBox);
                    }
                    // Result data part 2: OFIQ::FaceImageQualityAssessment::qAssessments (OFIQ::QualityAssessments)
                    {
                        OFIQ::QualityAssessments& qAssessments = assessment.qAssessments;
                        writer.write_scalar((uint16_t)qAssessments.size());
                        for (const auto& [measure_id, measure_result] : qAssessments)
                        {
                            // Unused: // #include <magic_enum.hpp> // auto name = static_cast<std::string>(magic_enum::enum_name(measure_id));
                            double rawScore = measure_result.rawScore;
                            double scalarScore = measure_result.scalar;
                            if (measure_result.code != QualityMeasureReturnCode::Success)
                                scalarScore = -1;
                            static_assert(sizeof(double) == 8, "Expected double to be 8 bytes.");
                            writer.write_scalar((int16_t)measure_id);
                            writer.write_scalar((uint8_t)measure_result.code);
                            writer.write_scalar(scalarScore);
                            writer.write_scalar(rawScore);
                        }
                    }
                    // Result data part 3: OFIQ_LIB::Session::getDetectedFaces() (std::vector<OFIQ::BoundingBox>)
                    {
                        std::vector<BoundingBox> detectedFaces = session.getDetectedFaces();
                        writer.write_scalar((uint16_t)detectedFaces.size());
                        for (BoundingBox& boundingBox : detectedFaces)
                            writer.write_ofiq_bounding_box(boundingBox);
                    }
                    // Result data part 4: OFIQ_LIB::Session::getPose() (OFIQ::EulerAngle)
                    {
                        std::array<double, 3> pose = session.getPose();
                        writer.write_scalar((double)pose[0]);
                        writer.write_scalar((double)pose[1]);
                        writer.write_scalar((double)pose[2]);
                    }
                    // Result data part 5: OFIQ_LIB::Session::getLandmarks() (OFIQ::FaceLandmarks)
                    {
                        FaceLandmarks faceLandmarks = session.getLandmarks();
                        writer.write_ofiq_landmarks(faceLandmarks);
                    }
                    // Result data part 6: OFIQ_LIB::Session::getAlignedFaceLandmarks() (OFIQ::FaceLandmarks)
                    {
                        FaceLandmarks faceLandmarks = session.getAlignedFaceLandmarks();
                        writer.write_ofiq_landmarks(faceLandmarks);
                    }
                    // Result data part 7: OFIQ_LIB::Session::getAlignedFaceTransformationMatrix() (cv::Mat)
                    {
                        cv::Mat mat = session.getAlignedFaceTransformationMatrix();
                        writer.write_cv_mat(mat);
                    }
                    // Result data part 8: OFIQ_LIB::Session::getAlignedFace() (cv::Mat)
                    {
                        cv::Mat mat = session.getAlignedFace();
                        writer.write_cv_mat(mat);
                    }
                    // Result data part 9: OFIQ_LIB::Session::getAlignedFaceLandmarkedRegion() (cv::Mat)
                    {
                        cv::Mat mat = session.getAlignedFaceLandmarkedRegion();
                        writer.write_cv_mat(mat);
                    }
                    // Result data part 10: OFIQ_LIB::Session::getFaceParsingImage() (cv::Mat)
                    {
                        cv::Mat mat = session.getFaceParsingImage();
                        writer.write_cv_mat(mat);
                    }
                    // Result data part 11: OFIQ_LIB::Session::getFaceOcclusionSegmentationImage() (cv::Mat)
                    {
                        cv::Mat mat = session.getFaceOcclusionSegmentationImage();
                        writer.write_cv_mat(mat);
                    }
                }
            }
            // Send the results:
            zmq_send_message(socket, writer);
        } break;
        default: {
            cerr << "[OFIQ_zmq_app][WARNING] Ignoring invalid received command: " << command_type << endl;
        } break;
    }
    return 0;
}

int main(int argc, char* argv[])
{
    // ZeroMQ setup:
    void *context = zmq_ctx_new();
    void *responder = zmq_socket(context, ZMQ_REP);
    {
        const char* endpoint = "tcp://*:40411";
        int rc = zmq_bind(responder, endpoint);
        if (rc != 0) {
            cerr << "[OFIQ_zmq_app][ERROR] zmq_bind('" << endpoint << "'): " << zmq_strerror(errno) << " (errno " << errno << ")" << endl;
            return FAILURE;
        }
        cout << "[OFIQ_zmq_app][INFO] ZeroMQ server bound to: " << endpoint << endl;
    }

    // OFIQ: Get implementation pointer
    std::shared_ptr<Interface> implPtr = Interface::getImplementation();
    // OFIQ: Initialization
    {
        auto ret = implPtr->initialize("data", "ofiq_config.jaxn");
        if (ret.code != ReturnCode::Success)
        {
            cerr << "[OFIQ_zmq_app][ERROR] OFIQ initialization failed: " << ret.code << "." << endl
                << ret.info << endl;
            return FAILURE;
        }
    }
    cout << "[OFIQ_zmq_app][INFO] OFIQ initialized." << endl;

    // ZeroMQ server loop:
    std::vector<uint8_t> full_message_data;
    while (true) {
        // Await/Receive next message:
        {
            int return_code = zmq_receive_multipart_message(responder, full_message_data);
            if (return_code == 0) { // Regular shutdown due to receive timeout.
                cout << "[OFIQ_zmq_app][INFO] Shutdown due to receive timeout." << endl;
                return SUCCESS;
            }
            if (return_code < 0) // Shutdown due to unexpected error.
                return FAILURE;
        }

        // Process message:
        {
            int return_code = process_message(responder, implPtr, full_message_data);
            if (return_code == -1) {
                // Processing failed, notify requester:
                Writer writer;
                writer.write_header(255);
                zmq_send_message(responder, writer);
            }
            else if (return_code == 1) return SUCCESS;
        }
    }

    return SUCCESS; // Should not be reachable.
}
