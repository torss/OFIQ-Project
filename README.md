# Open Source Face Image Quality (OFIQ) - Unofficial ZeroMQ/Python fork

The intended purpose of this unofficial fork/branch is to make OFIQ more easily/efficiently usable from Python, and also to expose additional `OFIQ_LIB::Session` data.

**Quick start:** Build OFIQ as per usual, then see the documentation at the top of [python/ofiq_zmq.py](python/ofiq_zmq.py).

**Python dependencies:** The Python side ([python/ofiq_zmq.py](python/ofiq_zmq.py)) currently requires an installation of the ["fiqat" (FIQA Toolkit)](https://share.nbl.nislab.no/g03-03-sample-quality/face-image-quality-toolkit).
This requirement could be removed if desired, as it is mainly used to handle image loading.

**Added C++ dependencies:** The OFIQ build now requires `zeromq`, which should however be handled automatically by the Conan package manager (i.e. `zeromq/4.3.5` has been added to the [conan/conanfile.txt](conan/conanfile.txt)).
But note that the build now also requires C++20 (`CMAKE_CXX_STANDARD 20`), albeit only for `std::endian` in the new [OFIQlib/src/OFIQ_zmq_app.cpp](OFIQlib/src/OFIQ_zmq_app.cpp) file.

**Details & extensibility to other languages:** To facilitate the Python usage, this adds a C++ executable target with the file [OFIQlib/src/OFIQ_zmq_app.cpp](OFIQlib/src/OFIQ_zmq_app.cpp), which acts as a ZeroMQ server (`ZMQ_REP`) through which OFIQ can be used.
The single added client (`ZMQ_REQ`) is the Python module [python/ofiq_zmq.py](python/ofiq_zmq.py), which controls the start/stop of the `OFIQ_zmq_app` executable.
Technically other language adapters could use the `OFIQ_zmq_app` executable as well, but there currently are no plans to implement any.

**C++ file changes:** Some OFIQ C++ have been extended to expose `OFIQ_LIB::Session` data.
As a result, OpenCV is now required as dependency for applications that use this modified OFIQ version (e.g. [OFIQSampleApp.cpp](OFIQlib/src/OFIQSampleApp.cpp) which otherwise doesn't need OpenCV). This could be fixed if required, but there currently is no plan to do so, as this is meant as a modified standalone OFIQ fork/branch that allows Python usage.

**License:** Like the majority of the OFIQ code, the new Python file ([python/ofiq_zmq.py](python/ofiq_zmq.py)) and C++ file ([OFIQlib/src/OFIQ_zmq_app.cpp](OFIQlib/src/OFIQ_zmq_app.cpp)) use the MIT License (see the file headers).

**Miscellaneous:** Two [scripts/build.sh](scripts/build.sh) variants have been added ([build__no_iso_download.sh](scripts/build__no_iso_download.sh) & [build__only_build.sh](scripts/build__only_build.sh)), but these are unimportant helpers for C++ development purposes.
A [.gitignore](.gitignore) file has also been added.

The remainder of this README is the original content (except with nested headers).

## Open Source Face Image Quality (OFIQ)

The __OFIQ__ (Open Source Face Image Quality) is a software library for computing quality 
aspects of a facial image. OFIQ is written in the C/C++ programming language.
OFIQ is the reference implementation for the ISO/IEC 29794-5 international
standard; see [https://bsi.bund.de/dok/OFIQ-e](https://bsi.bund.de/dok/OFIQ-e).

### License
Before using __OFIQ__ or distributing parts of __OFIQ__ one should have a look
on OFIQ's license and the license of its dependencies: [LICENSE.md](LICENSE.md)
  
### Reference manual
The best way to get started, is to read OFIQ's reference manual: 
see [doc/refman.pdf](doc/refman.pdf). The manual contains a documentation on
how to compile and run __OFIQ__.
