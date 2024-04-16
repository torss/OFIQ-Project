"""OFIQ can be used via this Python module.

Note that parts of this Python module rely on the "fiqat" (FIQA Toolkit) available here:
<https://share.nbl.nislab.no/g03-03-sample-quality/face-image-quality-toolkit>

The OfiqZmq class acts as a client to the OFIQ_zmq_app C++ program (built from OFIQlib/src/OFIQ_zmq_app.cpp).
The start & shutdown of OFIQ_zmq_app is handled by the Python side,
but the path to the OFIQ build directory (or to the OFIQ_zmq_app directly) needs to be provided.

See the main() function at the bottom for a simple usage example.
This is meant to be used as a module, so your code could look like this:
  from ofiq_zmq import OfiqZmq
  # ...
  ofiq_zmq = OfiqZmq('my/local/path/OFIQ-Project')
  results1 = ofiq_zmq.process_image('test1.jpg')
  results2 = ofiq_zmq.process_image('test2.jpg')
  ofiq_zmq.shutdown()

Besides the main OFIQ::FaceImageQualityAssessment output data,
this also allows access to OFIQ_LIB::Session data, such as the landmarks or the aligned face image.
See the OFIQ documentation, the process_image output (a dictionary),
or the process_image function comments ("Result data part ...").

Descriptions of structures mirroring the OFIQ C++ side
are based on the OFIQ GitHub repository <https://github.com/BSI-OFIQ/OFIQ-Project>
at commit ae44e41d6796e29d3071d9e4f3321fec72f8abf6 (Tue Apr 9 15:10:40 2024 +0200).

License:
  MIT License

  Copyright (c) 2024 da/sec

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.

Author:
  Torsten Schlett
"""

# Standard imports:
from typing import Union
from typing import NamedTuple
from typing import Optional
from pathlib import Path
import platform
import subprocess
import io
import struct
from enum import IntEnum

# External imports:
import numpy as np
import zmq

# Toolkit import:
import fiqat


class CommandType(IntEnum):
  PING = 0
  SHUTDOWN = 1
  PROCESS_IMAGE = 2
  MESSAGE_PROCESSING_FAILED = 255


expected_message_format_version = 1

struct_type_fmts = {
    'uint8_t': '!B',
    'uint16_t': '!H',
    'uint32_t': '!I',
    'uint64_t': '!Q',
    'int8_t': '!b',
    'int16_t': '!h',
    'int32_t': '!i',
    'int64_t': '!q',
    'float': '!f',
    'double': '!d',
}
struct_type_sizes = {
    'uint8_t': 1,
    'uint16_t': 2,
    'uint32_t': 4,
    'uint64_t': 8,
    'int8_t': 1,
    'int16_t': 2,
    'int32_t': 4,
    'int64_t': 8,
    'float': 4,
    'double': 8,
}


class OfiqZmqException(Exception):
  pass


class OfiqQualityMeasure(IntEnum):
  """Enums presenting the measure labels."""
  UNIFIED_QUALITY_SCORE = 0x41
  """UnifiedQualityScore"""
  BACKGROUND_UNIFORMITY = 0x42
  """BackgroundUniformity"""
  ILLUMINATION_UNIFORMITY = 0x43
  """IlluminationUniformity"""
  LUMINANCE = -0x44
  """the common measure implementation for LuminanceMean, LuminanceVariance"""
  LUMINANCE_MEAN = 0x44
  """LuminanceMean"""
  LUMINANCE_VARIANCE = 0x45
  """LuminanceVariance"""
  UNDER_EXPOSURE_PREVENTION = 0x46
  """UnderExposurePrevention"""
  OVER_EXPOSURE_PREVENTION = 0x47
  """OverExposurePrevention"""
  DYNAMIC_RANGE = 0x48
  """DynamicRange"""
  SHARPNESS = 0x49
  """Sharpness"""
  COMPRESSION_ARTIFACTS = 0x4a
  """CompressionArtifacts"""
  NATURAL_COLOUR = 0x4b
  """NaturalColour"""
  SINGLE_FACE_PRESENT = 0x4c
  """SingleFacePresent"""
  EYES_OPEN = 0x4d
  """EyesOpen"""
  MOUTH_CLOSED = 0x4e
  """MouthClosed"""
  EYES_VISIBLE = 0x4f
  """EyesVisible"""
  MOUTH_OCCLUSION_PREVENTION = 0x50
  """MouthOcclusionPrevention"""
  FACE_OCCLUSION_PREVENTION = 0x51
  """FaceOcclusionPrevention"""
  INTER_EYE_DISTANCE = 0x52
  """InterEyeDistance"""
  HEAD_SIZE = 0x53
  """HeadSize"""
  CROP_OF_THE_FACE_IMAGE = -0x54
  """CropOfTheFaceImage: common measure for {Left,Right,Up,Down}wardCropOfTheFaceImage"""
  LEFTWARD_CROP_OF_THE_FACE_IMAGE = 0x54
  """LeftwardCropOfTheFaceImage"""
  RIGHTWARD_CROP_OF_THE_FACE_IMAGE = 0x55
  """RightwardCropOfTheFaceImage"""
  DOWNWARD_CROP_OF_THE_FACE_IMAGE = 0x56
  """DownwardCropOfTheFaceImage"""
  UPWARD_CROP_OF_THE_FACE_IMAGE = 0x57
  """UpwardCropOfTheFaceImage"""
  HEAD_POSE = -0x58
  """HeadPose"""
  HEAD_POSE_YAW = 0x58
  """HeadPoseYaw"""
  HEAD_POSE_PITCH = 0x59
  """HeadPosePitch"""
  HEAD_POSE_ROLL = 0x5a
  """HeadPoseRoll"""
  EXPRESSION_NEUTRALITY = 0x5b
  """ExpressionNeutrality"""
  NO_HEAD_COVERINGS = 0x5c
  """NoHeadCoverings"""
  NOT_SET = -1
  """unknown measure"""


class OfiqFaceDetectorType(IntEnum):
  """Enum describing the different face detector implementations."""
  OPENCVSSD = 0
  """face detector based on the ssd implementation in opencv."""
  NOT_SET = 1  # NotSet
  """unknown face detector"""


class OfiqBoundingBox(NamedTuple):
  """Data structure for descibing bounding boxes, e.g. the face region of the faces found by a face detector."""
  xleft: int  # int16_t
  """leftmost point on head, typically subject's right ear value must be on [0, imageWidth-1]"""
  ytop: int  # int16_t
  """high point of head, typically top of hair; value must be on [0, imageHeight-1]"""
  width: int  # int16_t
  """bounding box width"""
  height: int  # int16_t
  """bounding box height"""
  face_detector_type: OfiqFaceDetectorType  # (uint8_t)
  """Description of the face detector used."""


class OfiqQualityMeasureReturnCode(IntEnum):
  """Return codes for QualityMeasureResult"""
  SUCCESS = 0
  """Success"""
  FAILURE_TO_ASSESS = 1
  """Unable to assess a quality measure"""
  NOT_INITIALIZED = 2
  """Quality measure is not initialized"""


class OfiqQualityMeasureResult(NamedTuple):
  """Data structure to handle the results of a quality measure."""
  status_code: OfiqQualityMeasureReturnCode
  """Return status code."""
  scalar_score: int  # double (converted to an int on the Python side)
  """A scalar value from the interval [0,100] Higher values mean higher quality. A value of -1.0 indicates a failed
  attempt to calculate a quality score or the value is unassigned."""
  raw_score: float  # double
  """Raw value as computed by the quality measure implementation."""


class OfiqLandmarkType(IntEnum):
  """Enum describing the different implementations of landmarks."""
  LM_98 = 0
  """Landmarks extracted with the adnet detector."""
  NOT_SET = 1
  """used for iterating through the enums."""


class OfiqLandmarkPoint(NamedTuple):
  """Data structure to describe the x and y coordinate of a landmark."""
  x: int  # int16_t # pylint: disable=invalid-name
  """x - coordinate"""
  y: int  # int16_t # pylint: disable=invalid-name
  """y - coordinate"""


class OfiqFaceLandmarks(NamedTuple):
  """Data structure for storing facial landmarks."""
  landmark_type: OfiqLandmarkType
  landmarks: list[OfiqLandmarkPoint]


class Writer:

  __slots__ = ['file']

  def __init__(self) -> None:
    self.file = io.BytesIO()

  def write_scalar(self, struct_type: str, value: Union[int, float]):
    struct_fmt = struct_type_fmts[struct_type]
    self.file.write(struct.pack(struct_fmt, value))

  def write_header(self, command_type: CommandType):
    self.write_scalar('uint64_t', expected_message_format_version)
    self.write_scalar('uint8_t', command_type)

  def write_image(self, input_image: fiqat.InputImage):
    image = fiqat.load_input_image(input_image, fiqat.ImageChannelType.RGB).image
    image = np.ascontiguousarray(image)
    self.write_scalar('uint16_t', image.shape[1])  # width
    self.write_scalar('uint16_t', image.shape[0])  # height
    self.file.write(image.data)

  def send(self, socket: zmq.Socket):
    buffer = self.file.getbuffer()
    socket.send(buffer)


class Reader:

  __slots__ = ['file', 'msg_bytes_len']

  def __init__(self, socket: zmq.Socket) -> None:
    msg_bytes = socket.recv()
    self.msg_bytes_len = len(msg_bytes)
    self.file = io.BytesIO(msg_bytes)

  def read_scalar(self, struct_type: str) -> Union[int, float]:
    struct_fmt = struct_type_fmts[struct_type]
    struct_size = struct_type_sizes[struct_type]
    return struct.unpack(struct_fmt, self.file.read(struct_size))[0]

  def read_header(self, expected_command_type: Optional[CommandType] = None) -> CommandType:
    message_format_version = self.read_scalar('uint64_t')
    if message_format_version != expected_message_format_version:
      raise OfiqZmqException('Unexpected message_format_version', message_format_version,
                             ('expected', expected_message_format_version))
    command_type_int = self.read_scalar('uint8_t')
    command_type = CommandType(command_type_int)
    if (expected_command_type is not None) and (command_type != expected_command_type):
      raise OfiqZmqException('Unexpected CommandType', command_type, ('expected', expected_command_type))
    return command_type

  def read_ofiq_bounding_box(self) -> OfiqBoundingBox:
    xleft = self.read_scalar('int16_t')
    ytop = self.read_scalar('int16_t')
    width = self.read_scalar('int16_t')
    height = self.read_scalar('int16_t')
    face_detector = self.read_scalar('uint8_t')
    face_detector_type = OfiqFaceDetectorType(face_detector)
    return OfiqBoundingBox(xleft, ytop, width, height, face_detector_type)

  def read_ofiq_quality_assessments(self) -> dict:
    quality_assessments = {}
    count = self.read_scalar('uint16_t')
    for _ in range(count):
      measure_id_int = self.read_scalar('int16_t')
      measure_id = OfiqQualityMeasure(measure_id_int)
      status_code_int = self.read_scalar('uint8_t')
      status_code = OfiqQualityMeasureReturnCode(status_code_int)
      scalar_score_double = self.read_scalar('double')
      scalar_score_int = int(scalar_score_double)
      if scalar_score_int != int(round(scalar_score_double)):
        raise OfiqZmqException(
            'Reader.read_ofiq_quality_assessments unexpectedly received a scalar_score_double'
            ' that does not represent an integer', ('scalar_score_double', scalar_score_double))
      raw_score = self.read_scalar('double')
      quality_assessments[measure_id] = OfiqQualityMeasureResult(status_code, scalar_score_int, raw_score)
    return quality_assessments

  def read_ofiq_detected_faces(self) -> list[OfiqBoundingBox]:
    detected_faces = []
    count = self.read_scalar('uint16_t')
    for _ in range(count):
      detected_faces.append(self.read_ofiq_bounding_box())
    return detected_faces

  def read_ofiq_pose(self) -> tuple:
    angle1 = self.read_scalar('double')
    angle2 = self.read_scalar('double')
    angle3 = self.read_scalar('double')
    return (angle1, angle2, angle3)

  def read_ofiq_landmarks(self) -> OfiqFaceLandmarks:
    landmark_type_int = self.read_scalar('uint8_t')
    landmark_type = OfiqLandmarkType(landmark_type_int)
    count = self.read_scalar('uint32_t')
    landmark_points = []
    for _ in range(count):
      point_x = self.read_scalar('int16_t')
      point_y = self.read_scalar('int16_t')
      landmark_points.append(OfiqLandmarkPoint(point_x, point_y))
    return OfiqFaceLandmarks(landmark_type, landmark_points)

  def read_cv_mat(self) -> Optional[np.ndarray]:
    cols = self.read_scalar('int32_t')
    if cols == -1:  # Check for the unsupported-cv::Mat marker.
      return None
    rows = self.read_scalar('int32_t')
    channels = self.read_scalar('int32_t')
    depth = self.read_scalar('uint8_t')
    shape = (rows, cols, channels)
    dtype = None
    if depth == 0:  # CV_8U
      dtype = np.uint8
      struct_type = 'uint8_t'
    elif depth == 1:  # CV_8S
      dtype = np.int8
      struct_type = 'int8_t'
    elif depth == 2:  # CV_16U
      dtype = np.uint16
      struct_type = 'uint16_t'
    elif depth == 3:  # CV_16S
      dtype = np.int16
      struct_type = 'int16_t'
    elif depth == 4:  # CV_32S
      dtype = np.int32
      struct_type = 'int32_t'
    elif depth == 5:  # CV_32F
      dtype = np.float32
      struct_type = 'float'
    elif depth == 6:  # CV_64F
      dtype = np.float64
      struct_type = 'double'
    else:
      raise OfiqZmqException('Reader.read_cv_mat - Invalid depth value', depth, ('misc: shape', shape))
    struct_fmt = struct_type_fmts[struct_type]
    struct_size = struct_type_sizes[struct_type]
    file = self.file
    result = np.zeros(shape, dtype)
    for row in range(rows):
      for col in range(cols):
        for channel in range(channels):
          result[row, col, channel] = struct.unpack(struct_fmt, file.read(struct_size))[0]
    return result

  def read_cv_mat_as_image(self) -> Optional[fiqat.TypedNpImage]:
    image = self.read_cv_mat()
    if image is None:
      return None
    if (len(image.shape) != 3) and (len(image.shape) != 2):
      raise OfiqZmqException('Reader.read_cv_mat_as_image - Unexpected image shape length', image.shape, image)
    image_channel_type = None
    if (len(image.shape) < 3) or (image.shape[2] == 1):
      image_channel_type = fiqat.ImageChannelType.GRAY
    elif image.shape[2] == 3:
      # Expect BGR order as loaded by cv2 by default.
      image_channel_type = fiqat.ImageChannelType.BGR
    else:
      raise OfiqZmqException('Reader.read_cv_mat_as_image - Invalid image shape in terms of channel count', image.shape,
                             image)
    return fiqat.TypedNpImage(image, image_channel_type)

  def check_end(self, command_type: CommandType):
    remaining_bytes = self.msg_bytes_len - self.file.tell()
    if remaining_bytes != 0:
      fiqat.term.cprint(
          'WARNING - OFIQ-ZeroMQ Reader.check_end - Unexpected leftover data at the end of the message: '
          f'{remaining_bytes} bytes ({str(command_type)})', 'yellow')


def get_ofiq_dir(input_path: Path) -> Path:
  relative_config_path = 'data/ofiq_config.jaxn'
  path = input_path
  while True:
    if (path / relative_config_path).is_file():
      return path
    if path == path.parent:
      raise OfiqZmqException('get_ofiq_dir failed', ('input', input_path))
    path = path.parent


def get_executable_path_from_ofiq_dir(ofiq_dir: Path) -> Path:
  if platform.system() == 'Windows':
    try_relative_paths = [
        Path('install_x86_64/Release/bin/OFIQ_zmq_app.exe'),
        Path('install_x86_64/Release/bin/OFIQ_zmq_app'),
    ]
  else:
    try_relative_paths = [Path('install_x86_64_linux/Release/bin/OFIQ_zmq_app')]
  for relative_path in try_relative_paths:
    ofiq_zmq_app_path = ofiq_dir / relative_path
    if ofiq_zmq_app_path.is_file():
      return ofiq_zmq_app_path
  raise OfiqZmqException('get_executable_path_from_ofiq_dir failed', ('input', ofiq_dir))


class OfiqZmq:

  def __init__(self, ofiq_path: Union[Path, str]) -> None:
    """``ofiq_path`` can either be
    - the OFIQ build directory (containing /data/ofiq_config.jaxn)
    - or the OFIQ_zmq_app executable path.
    """
    ofiq_path = Path(ofiq_path).resolve()
    if ofiq_path.is_dir():
      ofiq_dir = get_ofiq_dir(ofiq_path)
      ofiq_zmq_app_path = get_executable_path_from_ofiq_dir(ofiq_dir)
    else:
      ofiq_zmq_app_path = ofiq_path
      ofiq_dir = get_ofiq_dir(ofiq_zmq_app_path)
    self.ofiq_dir = ofiq_dir
    self.ofiq_zmq_app_path = ofiq_zmq_app_path
    self.address = 'tcp://127.0.0.1:40411'
    self.context = None
    self.socket = None
    self.internal_image_id_counter = 0

  def start(self):
    """Initializes ZeroMQ and starts the OFIQ C++ server part if required.
    This doesn't have to be called manually, as ``process_image`` will call this anyway.
    """
    expect_server = False
    if self.context is None:
      self.context = zmq.Context()
      self.socket = self.context.socket(zmq.REQ)
      self.socket.connect(self.address)
    else:
      expect_server = True
    # - #
    if (not expect_server) or (not self._ping()):
      subprocess.Popen([str(self.ofiq_zmq_app_path)], cwd=self.ofiq_dir)

  def _ping(self, timeout_ms: int = 1000) -> bool:
    writer = Writer()
    writer.write_header(CommandType.PING)
    writer.send(self.socket)
    # - #
    ping_response = False
    try:
      self.socket.set(zmq.RCVTIMEO, timeout_ms)
      reader = Reader(self.socket)
      reader.read_header(CommandType.PING)
      reader.check_end(CommandType.PING)
      ping_response = True
    except zmq.ZMQError:
      ping_response = False
      # Close the old socket and create a new one to discard the "send" above.
      # Cf. <http://wiki.zeromq.org/area:faq>:
      # "There is no explicit command for flushing a specific message or all messages from the message queue. You may
      # set ZMQ_LINGER to 0 and close the socket to discard any unsent messages."
      self.socket.close(linger=0)
      self.socket = self.context.socket(zmq.REQ)
      self.socket.connect(self.address)
    self.socket.set(zmq.RCVTIMEO, -1)
    return ping_response

  def process_image(self, input_image: Union[fiqat.InputImage, str]) -> Optional[dict]:
    if isinstance(input_image, str):
      input_image = Path(input_image)
    # - #
    self.start()
    # - #
    self.internal_image_id_counter += 1
    if self.internal_image_id_counter > 4_294_967_295:  # uint32_t
      self.internal_image_id_counter = 0
    internal_image_id = self.internal_image_id_counter
    # - #
    writer = Writer()
    writer.write_header(CommandType.PROCESS_IMAGE)
    writer.write_scalar('uint32_t', internal_image_id)
    writer.write_image(input_image)
    writer.send(self.socket)
    # - #
    reader = Reader(self.socket)
    reader.read_header(CommandType.PROCESS_IMAGE)
    response_image_id = reader.read_scalar('uint32_t')
    if internal_image_id != response_image_id:
      raise OfiqZmqException('internal_image_id != response_image_id', internal_image_id, response_image_id)
    processing_success = reader.read_scalar('uint8_t')
    if processing_success == 0:
      results = None
    else:
      results = {}
      # Result data part 1: OFIQ::FaceImageQualityAssessment::boundingBox (OFIQ::BoundingBox)
      results['bounding_box'] = reader.read_ofiq_bounding_box()
      # Result data part 2: OFIQ::FaceImageQualityAssessment::qAssessments (OFIQ::QualityAssessments)
      results['quality_assessments'] = reader.read_ofiq_quality_assessments()
      # Result data part 3: OFIQ_LIB::Session::getDetectedFaces() (std::vector<OFIQ::BoundingBox>)
      results['detected_faces'] = reader.read_ofiq_detected_faces()
      # Result data part 4: OFIQ_LIB::Session::getPose() (OFIQ::EulerAngle)
      results['pose'] = reader.read_ofiq_pose()
      # Result data part 5: OFIQ_LIB::Session::getLandmarks() (OFIQ::FaceLandmarks)
      results['face_landmarks'] = reader.read_ofiq_landmarks()
      # Result data part 6: OFIQ_LIB::Session::getAlignedFaceLandmarks() (OFIQ::FaceLandmarks)
      results['aligned_face_landmarks'] = reader.read_ofiq_landmarks()
      # Result data part 7: OFIQ_LIB::Session::getAlignedFaceTransformationMatrix() (cv::Mat)
      results['aligned_face_transformation_matrix'] = reader.read_cv_mat()
      # Result data part 8: OFIQ_LIB::Session::getAlignedFace() (cv::Mat)
      results['aligned_face'] = reader.read_cv_mat_as_image()
      # Result data part 9: OFIQ_LIB::Session::getAlignedFaceLandmarkedRegion() (cv::Mat)
      results['aligned_face_landmarked_region'] = reader.read_cv_mat_as_image()
      # Result data part 10: OFIQ_LIB::Session::getFaceParsingImage() (cv::Mat)
      results['face_parsing_image'] = reader.read_cv_mat_as_image()
      # Result data part 11: OFIQ_LIB::Session::getFaceOcclusionSegmentationImage() (cv::Mat)
      results['face_occlusion_segmentation_image'] = reader.read_cv_mat_as_image()
    reader.check_end(CommandType.PROCESS_IMAGE)
    return results

  def shutdown(self):
    """This will send a shutdown command to the OFIQ C++ side, and destroy the ZeroMQ context.
    Note that the OFIQ C++ side will shutdown automatically after a timeout, so calling this isn't strictly necessary.
    """
    self.start()
    # - #
    writer = Writer()
    writer.write_header(CommandType.SHUTDOWN)
    writer.send(self.socket)
    # - #
    reader = Reader(self.socket)
    reader.read_header(CommandType.SHUTDOWN)
    reader.check_end(CommandType.SHUTDOWN)
    # - #
    self.context.destroy()
    self.context = None
    self.socket = None


def main():
  # NOTE This module is meant to be used by your custom code via the OfiqZmq class.
  #      This function is just a small usage example that can be used to test whether the module functions.

  import sys  # pylint: disable=import-outside-toplevel

  if len(sys.argv) != 3:
    print('Usage: python ofiq_zmq.py <path to OFIQ build dir or OFIQ_zmq_app executable> <path to a test input image>')
    return

  ofiq_path = Path(sys.argv[1])
  input_image_path = Path(sys.argv[2])

  ofiq_zmq = OfiqZmq(ofiq_path)
  print('OFIQ dir:', ofiq_zmq.ofiq_dir)
  print('OFIQ_zmq_app executable:', ofiq_zmq.ofiq_zmq_app_path)

  results = ofiq_zmq.process_image(input_image_path)

  ofiq_zmq.shutdown()  # Should be called after all images have been processed. Not strictly necessary.

  if results is None:
    print('ofiq_zmq.process_image failed completely for the input (empty output)')
  else:
    print('Quality assessment output (scalar scores):')
    for measure_id, measure_result in results['quality_assessments'].items():
      print(f'- {str(measure_id):>52}: {measure_result.scalar_score:>3} ({str(measure_result.status_code)})')

  return results


if __name__ == '__main__':
  main()
