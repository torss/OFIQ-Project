#!/bin/bash
build_dir=build/build_linux
install_dir=install_x86_64_linux
use_conan=ON
download=ON
config=Release
os=linux
useSonarWrapper=false
sonarWrapper="${SONARQUBE_WRAPPER_PATH:-.sonar}/build-wrapper-linux-x86-64"

while [ -n "$1" ]
do  
if [ "$1" = "--no-conan" ]; then
    use_conan=OFF
elif [ "$1" = "--no-download" ]; then
    download=OFF
elif [ "$1" = "--debug" ]; then
    config=Debug
elif [ "$1" = "--use-sonar-wrapper" ]; then
    useSonarWrapper=true
    echo "The following sonar wrapper will be used: $sonarWrapper"
elif [ "$1" = "--os" ]; then
    shift
    if [ "$1" = "macos" ]; then
        build_dir=build/build_mac
        install_dir=install_arm64_mac
        os=macos
    elif [ "$1" = "linux-arm64" ]; then
        build_dir=build/build_linux_arm64
        install_dir=install_arm64_linux
        os=linux_arm64
    else
        echo "$1" is a not a supported OS
        exit
    fi
else
    echo "$1" is not a supported argument
    exit
fi
shift
done

if [ "$use_conan" = "ON" ]
then
    rm -rf ../build/conan

    export OFIQLIB_CONAN_DIR=../conan
    export CONAN_FILE=conanfile.txt

    echo using conan file ${CONAN_FILE}

    if [ "$config" = "Release" ]
    then
        conan install ${OFIQLIB_CONAN_DIR}/${CONAN_FILE} \
            --build missing \
            --profile:build ${OFIQLIB_CONAN_DIR}/conan_profile_release_${os}.txt \
            --profile:host ${OFIQLIB_CONAN_DIR}/conan_profile_release_${os}.txt \
            --output-folder=../build/conan \
            -g CMakeDeps \
            -g CMakeToolchain
    else
        conan install ${OFIQLIB_CONAN_DIR}/${CONAN_FILE} \
        --build missing \
        --profile:build ${OFIQLIB_CONAN_DIR}/conan_profile_debug_${os}.txt \
        --profile:host ${OFIQLIB_CONAN_DIR}/conan_profile_debug_${os}.txt \
        --output-folder=../build/conan \
        -g CMakeDeps \
        -g CMakeToolchain
    fi
else
    if [ "$download" = "ON" ]
    then
        cmake -P ../cmake/DownloadExternalSourceCode.cmake
    fi
    # build opencv
    cd ../extern/opencv-4.5.5
    cmake -S ./ -B build -DBUILD_LIST=core,calib3d,imgcodecs,improc,dnn,ml \
    -DBUILD_opencv_apps=OFF -DBUILD_opencv_java=OFF -DBUILD_opencv_python=OFF -DWITH_FFMPEG=OFF -DWITH_TIFF=OFF -DWITH_WEBP=OFF -DWITH_IPP=OFF \
    -DWITH_OPENCL=OFF -DWITH_LAPACK=OFF -DWITH_QUIRC=OFF \
    -DBUILD_ZLIB=ON -DWITH_ZLIB=ON \
    -DBUILD_PNG=ON -DWITH_PNG=ON \
    -DBUILD_JPEG=ON -DWITH_JPEG=ON \
    -DBUILD_OPENEXR=OFF -DWITH_OPENEXR=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DWITH_ADE=OFF -DWITH_CAROTENE=OFF \
    -DCMAKE_INSTALL_PREFIX=./build/install || exit 1
    cmake --build build --config $config --target install -j 8 || exit 1
    # build gtest
    cd ../googletest
    cmake -S ./ -B build -DBUILD_GMOCK=OFF -DINSTALL_GTEST=OFF -DBUILD_SHARED_LIBS=ON || exit 1
    cmake --build build/googletest --config $config || exit 1
    # build onnxruntime
    cd ../onnxruntime
    sh ./build.sh --config $config --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --update --build || exit 1
    cd ..
fi

echo "Attempting to build the real implementation"

cd ../
echo "Removing $build_dir"
rm -r $build_dir

echo "Generating build files"
cmake -S ./ -B $build_dir -DCMAKE_INSTALL_PREFIX=$install_dir -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DDOWNLOAD_ONNX=$use_conan -DUSE_CONAN=$use_conan -DOS=$os -DCMAKE_BUILD_TYPE=$config -DDOWNLOAD_MODELS_AND_IMAGES=$download

cmakeBuildCommand="cmake --build $build_dir --target install -j 8"
if [ "$useSonarWrapper" = "true" ]
then
    $sonarWrapper --out-dir bw-output $cmakeBuildCommand
else
    $cmakeBuildCommand
fi

echo "Building finished"
