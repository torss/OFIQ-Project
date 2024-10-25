# Open Source Face Image Quality (OFIQ) - Unofficial ZeroMQ/Python fork

The intended purpose of this unofficial fork/branch is to make OFIQ more easily/efficiently usable from Python, and also to expose additional `OFIQ_LIB::Session` data.
Note that this is currently mainly meant for internal use, so the documentation is rather minimal and there are some caveats regarding the dependencies (see blow).

**Quick start:** Build OFIQ as per usual, then see the documentation at the top of [python/ofiq_zmq.py](python/ofiq_zmq.py).

**Python dependencies:** The Python side ([python/ofiq_zmq.py](python/ofiq_zmq.py)) currently requires an installation of the [FIQA Toolkit "fiqat"](https://share.nbl.nislab.no/g03-03-sample-quality/face-image-quality-toolkit).
This requirement could be removed if desired, as it is mainly used to handle image loading.
Besides that it requires the packages [numpy](https://pypi.org/project/numpy/) (tested version `1.23.5`) and [pyzmq](https://pypi.org/project/pyzmq/) (tested version `25.0.2`).
The tested Python version is `3.9.16`.

**Added C++ dependencies:** The OFIQ build now requires [ZeroMQ](https://zeromq.org/) (specifically the [C version, i.e. "libzmq"](https://zeromq.org/languages/c/#libzmq)), which should however be handled automatically by the Conan package manager (i.e. [zeromq/4.3.5](https://conan.io/center/recipes/zeromq?version=4.3.5) has been added to the [conan/conanfile.txt](conan/conanfile.txt)).
But note that the build now also requires C++20 (`CMAKE_CXX_STANDARD 20`), albeit only for [std::endian](https://en.cppreference.com/w/cpp/types/endian) in the new [OFIQlib/src/OFIQ_zmq_app.cpp](OFIQlib/src/OFIQ_zmq_app.cpp) file.

**Details & extensibility to other languages:** To facilitate the Python usage, this adds a C++ executable target with the file [OFIQlib/src/OFIQ_zmq_app.cpp](OFIQlib/src/OFIQ_zmq_app.cpp), which acts as a ZeroMQ server (`ZMQ_REP`) through which OFIQ can be used.
The single added client (`ZMQ_REQ`) is the Python module [python/ofiq_zmq.py](python/ofiq_zmq.py), which controls the start/stop of the `OFIQ_zmq_app` executable.
Technically other language adapters could use the `OFIQ_zmq_app` executable as well, but there currently are no plans to implement any.

**C++ file changes:** Some OFIQ C++ files have been extended to expose `OFIQ_LIB::Session` data.
As a result, OpenCV is now required as a dependency for applications that use this modified OFIQ version (e.g. [OFIQSampleApp.cpp](OFIQlib/src/OFIQSampleApp.cpp) which otherwise doesn't need OpenCV). This could be fixed if required, but there currently is no plan to do so, as this is meant as a modified standalone OFIQ fork/branch that allows Python usage.

**License:** Like the majority of the OFIQ code, the new Python file ([python/ofiq_zmq.py](python/ofiq_zmq.py)) and C++ file ([OFIQlib/src/OFIQ_zmq_app.cpp](OFIQlib/src/OFIQ_zmq_app.cpp)) use the MIT License (see the file headers).

**Miscellaneous:** One [scripts/build.sh](scripts/build.sh) variant has been added ([build__only_build.sh](scripts/build__only_build.sh)), but that's an unimportant helper for C++ development purposes.
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
  
### Getting started
For a tutorial on how to compile and operate OFIQ, see [here](BUILD.md).

### Reference manual
A full documentation of __OFIQ__ including compilation, configuration and a comprehensive doxygen documentation of 
the C/C++ API is contained in the reference manual:
see [doc/refman.pdf](doc/refman.pdf).

