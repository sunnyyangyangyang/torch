%global pkgvers 0
%global scdate0 20260103
%global schash0 b3713dc2017e8dd6a2b090acac2447627093cf77
%global branch0 main
%global source0 https://github.com/pytorch/pytorch.git

%global sshort0 %{expand:%%{lua:print(('%{schash0}'):sub(1,8))}}

%global vcu_maj 13
%global vcu_min 1

# features
%define use_dnnl    0
%define use_magma   0
# ext libs
%define ext_fmt     0
%define ext_onnx    1
%define ext_gloo    0
%define ext_sleef   0
%define ext_kineto  1
%define ext_xnnpack 1
# flash attention
%define use_flashat 1
%define use_meffatt 1

Name:           pytorch
Version:        %(curl -s "https://raw.githubusercontent.com/pytorch/pytorch/%{schash0}/version.txt" | sed 's|.[a-z,A-Z]||')
Release:        %{scdate0}.%{pkgvers}.git%{sshort0}.cu%{vcu_maj}_%{vcu_min}%{?dist}
Summary:        PyTorch Neural Network Package 
License:        BSD

URL:            https://pytorch.org

Patch1:         pytorch-C.patch
Patch2:         pytorch-clang.patch

BuildRequires:  cuda-glibc-patch
BuildRequires:  mold
BuildRequires:  ninja-build
BuildRequires:  git doxygen cmake cmake-rpm-macros python3-devel pybind11-devel
BuildRequires:  python3-typing-extensions python3-pyyaml python3-setuptools
BuildRequires:  python3-wheel python3-pybind11 python3-six python3-numpy json-devel
BuildRequires:  cpuinfo-devel psimd-devel qnnpack-devel cutlass-devel gemmlowp-devel
BuildRequires:  mesa-libGLU-devel ocl-icd-devel libuv-devel rdma-core-devel miniz-devel
BuildRequires:  nnpack-devel gmp-devel mpfr-devel neon2sse-devel eigen3-devel >= 3.3.9
BuildRequires:  tensorpipe-devel fp16-devel fxdiv-devel zeromq-devel numactl-devel
BuildRequires:  glog-devel gflags-devel openblas-openmp pthreadpool-devel
BuildRequires:  foxi-devel snappy-devel openblas-devel libzstd-devel
BuildRequires:  openssl-devel fftw-devel flatbuffers-devel /usr/bin/flatc
BuildRequires:  protobuf-compat-devel >= 3.21
BuildRequires:  protobuf-compat-compiler >= 3.21

%if 0%{?rhel} == 8
BuildRequires:  python3-dataclasses
%endif

%ifnarch ppc64le
BuildRequires:  asmjit-devel
%endif

%ifarch x86_64
BuildRequires:  fbgemm-devel
%endif

%if %{use_dnnl}
BuildRequires:  onednn-devel ideep-devel
%endif

%if %{ext_fmt}
BuildRequires:  fmt-devel
%endif

%if %{ext_onnx}
BuildRequires:  onnx-devel onnx-optimizer-devel
%endif

%if %{ext_sleef}
BuildRequires:  sleef-devel
%endif

%if %{ext_kineto}
BuildRequires:  kineto-devel
%endif

%if %{ext_xnnpack}
BuildRequires:  xnnpack-devel xnnpack-static
%endif


%define have_cuda 1
%define have_tensorrt 0
%define have_cuda_gcc 1

%global toolchain gcc

%define gpu_target_arch "12.0+PTX"

%global _lto_cflags %{nil}
%global debug_package %{nil}
%global __cmake_in_source_build 1
%undefine _hardened_build
%undefine _annotated_build
%undefine _find_debuginfo_dwz_opts
%undefine _missing_build_ids_terminate_build

%bcond_without cuda
%if %{without cuda}
%global have_cuda 0
%endif

%if "%{toolchain}" == "gcc"
BuildRequires:  gcc-c++
%else
BuildRequires:  clang
%endif

# Workaround ARM64/GCC14
%if 0%{?fedora} > 39
%ifarch aarch64
BuildRequires:  gcc13-c++
%endif
%endif

%if %{have_cuda}
%if "%{toolchain}" == "gcc"
%if %{have_cuda_gcc}
%if 0%{?fedora} > 34
BuildRequires:  cuda-gcc-c++
%endif
%endif
%endif
%if "%{toolchain}" == "clang"
%if 0%{?fedora} > 40
BuildRequires:  clang18
%else
%if 0%{?fedora} > 39
BuildRequires:  clang17
%else
BuildRequires:  clang16
%endif
%endif
%endif
BuildRequires:  cuda-nvcc-%{vcu_maj}-%{vcu_min}
BuildRequires:  cuda-nvtx-%{vcu_maj}-%{vcu_min}
BuildRequires:  cuda-cupti-%{vcu_maj}-%{vcu_min}
BuildRequires:  cuda-cudart-devel-%{vcu_maj}-%{vcu_min}
BuildRequires:  cuda-nvml-devel-%{vcu_maj}-%{vcu_min}
BuildRequires:  cuda-nvrtc-devel-%{vcu_maj}-%{vcu_min}
BuildRequires:  cuda-driver-devel-%{vcu_maj}-%{vcu_min}
BuildRequires:  cuda-profiler-api-%{vcu_maj}-%{vcu_min}
BuildRequires:  libcublas-devel-%{vcu_maj}-%{vcu_min}
BuildRequires:  libcufft-devel-%{vcu_maj}-%{vcu_min}
BuildRequires:  libcurand-devel-%{vcu_maj}-%{vcu_min}
BuildRequires:  libcusparse-devel-%{vcu_maj}-%{vcu_min}
BuildRequires:  libcusolver-devel-%{vcu_maj}-%{vcu_min}
BuildRequires:  libnvjitlink-devel-%{vcu_maj}-%{vcu_min}
BuildRequires:  libcufile-devel-%{vcu_maj}-%{vcu_min}
BuildRequires:  libnvshmem3-devel-cuda-%{vcu_maj}
BuildRequires:  libcudnn9-devel-cuda-%{vcu_maj}
BuildRequires:  libnccl-devel
%if %{ext_gloo}
BuildRequires:  gloo-devel
%endif
%if %{use_magma}
BuildRequires:  magma-devel
%endif
Requires:  cuda-cudart-%{vcu_maj}-%{vcu_min}
Requires:  cuda-nvrtc-%{vcu_maj}-%{vcu_min}
Requires:  cuda-nvtx-%{vcu_maj}-%{vcu_min}
Requires:  libcublas-%{vcu_maj}-%{vcu_min}
Requires:  libcufft-%{vcu_maj}-%{vcu_min}
Requires:  libcurand-%{vcu_maj}-%{vcu_min}
Requires:  libcusparse-%{vcu_maj}-%{vcu_min}
Requires:  libcusolver-%{vcu_maj}-%{vcu_min}
Requires:  libnvjitlink-%{vcu_maj}-%{vcu_min}
Requires:  libcufile-%{vcu_maj}-%{vcu_min}
Requires:  libnvshmem3-cuda-%{vcu_maj}
Requires:  libcudnn9-cuda-%{vcu_maj}
%endif

%if %{have_tensorrt}
BuildRequires:  libnvinfer-plugin-devel libnvonnxparsers-devel
%endif

%description
PyTorch is a python package that provides two high-level
features for Torch.

%package        devel
Summary:        Development files for pytorch
Requires:       %{name} = %{version}-%{release}

%description    devel
This package contains development files for pythorch.

%package        python3
Summary:        Python files for pytorch
Provides:       python%{python3_version}dist(torch) = %{version}
Requires:       %{name} = %{version}-%{release}
Requires:       %{name}-devel = %{version}-%{release}

%description    python3
This package contains python files for pythorch.


%prep
%setup -T -c -n %{name}
git clone --depth 1 -n -b %{branch0} %{source0} .
git fetch --depth 1 origin %{schash0}
git reset --hard %{schash0}
%if ! %{ext_fmt}
git submodule update --init --depth 1 third_party/fmt
%endif
%if ! %{ext_onnx}
git submodule update --init --depth 1 third_party/onnx
%endif
%if ! %{ext_sleef}
git submodule update --init --depth 1 third_party/sleef
%endif
%if ! %{ext_gloo}
git submodule update --init --depth 1 third_party/gloo
sed -i '1i #include <cstdint>' third_party/gloo/gloo/types.h
%endif
%if ! %{ext_kineto}
git submodule update --init --depth 1 third_party/kineto
%endif
%if ! %{ext_xnnpack}
git submodule update --init --depth 1 third_party/XNNPACK
%endif
git submodule update --init --depth 1 third_party/ittapi
git submodule update --init --depth 1 third_party/pocketfft
git submodule update --init --depth 1 third_party/cudnn_frontend
git submodule update --init --depth 1 third_party/opentelemetry-cpp
git submodule update --init --depth 1 third_party/cpp-httplib
git submodule update --init --depth 1 third_party/kleidiai
%if %{use_flashat}
git submodule update --init --depth 1 third_party/flash-attention
%endif
git submodule update --init --depth 1 third_party/mimalloc
git --no-pager log --format=fuller
#global _default_patch_fuzz 100
%patch -P 1 -p0 -b .python~
%if "%{toolchain}" == "clang"
%patch -P 2 -p1 -b .clang~
%endif

# no benchmarks
mkdir -p third_party/benchmark
mkdir -p third_party/googletest
touch third_party/benchmark/CMakeLists.txt
touch third_party/googletest/CMakeLists.txt

# python version
sed -i -e 's|VERSION_LESS 3.7)|VERSION_LESS 3.6)|g' cmake/Dependencies.cmake
sed -i -e 's|PY_MAJOR_VERSION == 3|PY_MAJOR_VERSION == 3 \&\& PY_MINOR_VERSION > 6|' torch/csrc/dynamo/eval_frame.c
%if %{ext_gloo}
# external gloo
sed -i '1i #include <cstdint>' 
sed -i -e 's|torch_cpu PUBLIC c10|torch_cpu PUBLIC c10 gloo gloo_cuda|' caffe2/CMakeLists.txt
%endif
# external qnnpack
sed -i -e 's|torch_cpu PUBLIC c10|torch_cpu PUBLIC c10 qnnpack|' caffe2/CMakeLists.txt
# external pybind11
sed -i -e 's|USE_SYSTEM_BIND11|USE_SYSTEM_PYBIND11|g' cmake/Dependencies.cmake

%if %{use_dnnl}
# external mkl-dnn
rm -rf cmake/Modules/FindMKLDNN.cmake
echo 'set(DNNL_USE_NATIVE_ARCH ${USE_NATIVE_ARCH})' > cmake/public/mkldnn.cmake
echo 'set(CAFFE2_USE_MKLDNN ON)' >> cmake/public/mkldnn.cmake
echo 'find_package(DNNL REQUIRED)' >> cmake/public/mkldnn.cmake
echo 'set(MKLDNN_FOUND ON)' >> cmake/public/mkldnn.cmake
echo 'add_library(caffe2::mkldnn ALIAS DNNL::dnnl)' >> cmake/public/mkldnn.cmake
# external dnnl
sed -i -e 's|torch_cpu PUBLIC c10|torch_cpu PUBLIC c10 dnnl|' caffe2/CMakeLists.txt
%endif

# external pthreadpool
rm -rf third_party/pthreadpool/*
touch third_party/pthreadpool/CMakeLists.txt

# openblas openmp first
sed -i -e 's|NAMES openblas|NAMES openblaso openblas|' cmake/Modules/FindOpenBLAS.cmake

# use external onnx
%if %{ext_onnx}
sed -i -e 's|Caffe2_DEPENDENCY_LIBS onnx_proto onnx|Caffe2_DEPENDENCY_LIBS onnx_proto onnx onnx_optimizer|' cmake/Dependencies.cmake
%endif

# external tensorpipe
mkdir -p third_party/tensorpipe
echo '' >> third_party/tensorpipe/CMakeLists.txt
sed -i '/add_dependencies(tensorpipe_agent tensorpipe)/d' caffe2/CMakeLists.txt
sed -i '/arget_compile_options_if_supported(tensorpipe_uv/d' cmake/Dependencies.cmake

# external nnpack
echo '' > cmake/External/nnpack.cmake
echo 'set(NNPACK_FOUND TRUE)' >> cmake/External/nnpack.cmake

# external cpuinfo
sed -i '/TARGET cpuinfo PROPERTY/d' cmake/Dependencies.cmake

# external fp16
sed -i '/APPEND Caffe2_DEPENDENCY_LIBS fp16/d' cmake/Dependencies.cmake

# external qnnpack
mkdir -p third_party/QNNPACK
echo '' >> third_party/QNNPACK/CMakeLists.txt
sed -i '/TARGET qnnpack PROPERTY/d' cmake/Dependencies.cmake
sed -i -e '/target_compile_options(qnnpack/d' cmake/Dependencies.cmake

# external psimd
mkdir -p third_party/psimd
echo '' >> third_party/psimd/CMakeLists.txt
sed -i '/pytorch_qnnpack PRIVATE psimd/d' aten/src/ATen/native/quantized/cpu/qnnpack/CMakeLists.txt

# external fxdiv
sed -i '/NOT TARGET fxdiv/,/endif/d' caffe2/CMakeLists.txt
sed -i '/torch_cpu PRIVATE fxdiv/d' caffe2/CMakeLists.txt
sed -i '/pytorch_qnnpack PRIVATE fxdiv/d' aten/src/ATen/native/quantized/cpu/qnnpack/CMakeLists.txt

# external fbgemm
mkdir -p third_party/fbgemm
echo '' > third_party/fbgemm/CMakeLists.txt
sed -i '/(TARGET fbgemm/d' cmake/Dependencies.cmake
sed -i -e '/if_supported(fbgemm/d' cmake/Dependencies.cmake
sed -i -e '/target_compile_definitions(fbgemm/d' cmake/Dependencies.cmake

# external asmjit
sed -i -e '/if_supported(asmjit/d' cmake/Dependencies.cmake

# external foxi
mkdir -p third_party/foxi
echo '' > third_party/foxi/CMakeLists.txt

# external kineto
%if %{ext_kineto}
sed -i '/if(NOT TARGET kineto)/,/endif()/d' cmake/Dependencies.cmake
sed -i 's|libkineto/include|libkineto/include\n/usr/include/kineto|' torch/CMakeLists.txt
sed -i 's|libkineto/include|libkineto/include\n/usr/include/kineto|' caffe2/CMakeLists.txt
%endif

# external fmt
%if %{ext_fmt}
sed -i 's|add_subdirectory(.*/fmt)|find_package(fmt REQUIRED)|g' cmake/Dependencies.cmake
sed -i '/fmt-header-only PROPERTIES/d' cmake/Dependencies.cmake
%endif

# external miniz
#sed -i '/miniz.c/d' caffe2/serialize/CMakeLists.txt

# external xnnpack
%if %{ext_xnnpack}
sed -i 's|or NOT microkernels-prod|OR NOT microkernels-prod|g' cmake/Dependencies.cmake
%endif

# external tensorrt
mkdir -p third_party/onnx-tensorrt
echo '' > third_party/onnx-tensorrt/CMakeLists.txt
sed -i '/nvonnxparser_static/d' cmake/Dependencies.cmake
sed -i 's|onnx_trt_library|nvonnxparser_static|g' cmake/Dependencies.cmake

# flatbuffers
rm -rf torch/csrc/jit/serialization/mobile_bytecode_generated.h
flatc --cpp --gen-mutable --scoped-enums \
      -o torch/csrc/jit/serialization \
      -c torch/csrc/jit/serialization/mobile_bytecode.fbs
echo '// @generated' >> torch/csrc/jit/serialization/mobile_bytecode_generated.h

# no cmake cuda locals
mv -f  cmake/Modules_CUDA_fix/FindCUDNN.cmake cmake/Modules
rm -rf cmake/Modules_CUDA_fix
find . -type d -name "FindCUDA" -exec rm -rf {} \;
sed -i -e '/install/{:a;/COMPONENT/bb;N;ba;:b;/Modules_CUDA_fix/d;}' CMakeLists.txt
sed -i -e 's|CMAKE_CUDA_FLAGS "-D|CMAKE_CUDA_FLAGS " -D|' CMakeLists.txt

# disable AVX2 / SVE2
#sed -i -e 's|AVX2_FOUND|AVX2_NONE_FOUND|g' cmake/Codegen.cmake
#sed -i -e 's|SVE_FOUND|SVE_NONE_FOUND|g' cmake/Codegen.cmake

# remove export deps
sed -i '/install(EXPORT Caffe2Targets/,/dev)/d' CMakeLists.txt

# systeminc
sed -i 's|SYSTEM ||g' c10/CMakeLists.txt
sed -i 's|SYSTEM ||g' torch/CMakeLists.txt
sed -i 's|SYSTEM ||g' caffe2/CMakeLists.txt
sed -i 's|BEFORE SYSTEM ||g' cmake/ProtoBuf.cmake
sed -i 's|AFTER SYSTEM ||g' cmake/Dependencies.cmake
sed -i 's|BEFORE SYSTEM ||g' cmake/Dependencies.cmake
sed -i 's|SYSTEM ||g' cmake/Dependencies.cmake

# gcc13
sed -i '1i #include <stdexcept>' c10/util/Registry.h
sed -i '1i #include <cstdint>' c10/core/DispatchKey.h
sed -i '1i #include <stdexcept>' torch/csrc/jit/runtime/logging.cpp
sed -i '1i #include <stdexcept>' torch/csrc/lazy/core/multi_wait.cpp
sed -i '1i #include "stdint.h"' torch/csrc/jit/passes/quantization/quantization_type.h
sed -i '1i #include <algorithm>' aten/src/ATen/native/DispatchStub.cpp

#python 3.x
sed -i 's|${PYTHON_INCLUDE_DIR}|${PYTHON_INCLUDE_DIR}\n/usr/include/python%{python3_version}|' torch/CMakeLists.txt

find cmake -name "select_compute_arch.cmake" -exec sed -i 's/\^(\[0-9\]\\./^([0-9]+\\./g' {} +

%build
# GCC14 issues ARM64
%if 0%{?fedora} > 39
%ifarch aarch64
export CXX=g++-13
%endif
%endif

mkdir build
pushd build
export ONNX_ML=1
export BUILD_SPLIT_CUDA=ON
export REL_WITH_DEB_INFO=0
export TORCH_NVCC_FLAGS="-DCUDA_HAS_FP16"
export PYTHON_EXECUTABLE="%{__python3}"
%if "%{toolchain}" == "clang"
%if 0%{?fedora} > 40
export CUDAHOSTCXX="%{_bindir}/clang++-18"
%else
%if 0%{?fedora} > 39
export CUDAHOSTCXX="%{_bindir}/clang++-17"
%else
export CUDAHOSTCXX="%{_bindir}/clang++-16"
%endif
%endif
%endif
%global optflags %(echo "%{optflags} -w -fpermissive -Wno-sign-compare -Wno-deprecated-declarations -Wno-nonnull -DEIGEN_HAS_CXX11_MATH=1" | sed 's|-g||')
# -DUSE_NATIVE_ARCH=ON
export LD_LIBRARY_PATH="/usr/local/cuda-%{vcu_maj}.%{vcu_min}/%{_lib}/"
%cmake .. -Wno-dev \
       -GNinja \
       -DCMAKE_SKIP_RPATH=ON \
       -DCMAKE_VERBOSE_MAKEFILE=OFF \
       -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
       -DCMAKE_NO_SYSTEM_FROM_IMPORTED=ON \
       -DCMAKE_SKIP_RULE_DEPENDENCY=ON \
       -DCMAKE_SUPPRESS_REGENERATION=ON \
       -DUSE_CCACHE=OFF \
       -DHAVE_SOVERSION=ON \
       -DUSE_NATIVE_ARCH=OFF \
       -DUSE_PRIORITIZED_TEXT_FOR_LD=OFF \
       -DUSE_DISTRIBUTED=ON \
       -DBUILD_TEST=OFF \
       -DBUILD_DOCS=OFF \
       -DBUILD_PYTHON=ON \
       -DBUILD_FUNCTORCH=ON \
       -DBUILD_BINARY=OFF \
       -DBUILD_BENCHMARK=OFF \
       -DBUILD_CUSTOM_PROTOBUF=OFF \
       -DBUILDING_WITH_TORCH_LIBS=ON \
       -DPYTHON_EXECUTABLE="%{__python3}" \
       -DPYBIND11_PYTHON_VERSION="%{python3_version}" \
       -DCAFFE2_LINK_LOCAL_PROTOBUF=OFF \
       -DONNX_ML=ON \
       -DUSE_GLOG=ON \
       -DUSE_GFLAGS=ON \
       -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=mold" \
       -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=mold" \
       -DCMAKE_MODULE_LINKER_FLAGS="-fuse-ld=mold" \
%if "%{toolchain}" == "gcc"
       -DUSE_OPENMP=ON \
%else
       -DUSE_OPENMP=OFF \
%endif
       -DUSE_KINETO=ON \
%if %{ext_onnx}
       -DUSE_SYSTEM_ONNX=ON \
%else
       -DUSE_SYSTEM_ONNX=OFF \
%endif
%if %{ext_gloo}
       -DUSE_SYSTEM_GLOO=ON \
%else
       -DUSE_SYSTEM_GLOO=OFF \
%endif
       -DUSE_SYSTEM_PYBIND11=ON \
       -DUSE_SYSTEM_EIGEN_INSTALL=ON \
%if %{have_cuda}
       -DUSE_CUDA=ON \
       -DUSE_CUDNN=ON \
       -DUSE_NVRTC=ON \
       -DUSE_CUPTI_SO=ON \
       -DUSE_SYSTEM_NCCL=ON \
       -DUSE_SYSTEM_NVTX=ON \
       -DUSE_CUDSS=OFF \
       -DUSE_CUFILE=OFF \
%if %{use_flashat}
       -DUSE_FLASH_ATTENTION=ON \
%else
       -DUSE_FLASH_ATTENTION=OFF \
%endif
%if %{use_meffatt}
       -DUSE_MEM_EFF_ATTENTION=ON \
%else
       -DUSE_MEM_EFF_ATTENTION=OFF \
%endif
       -DCMAKE_CUDA_FLAGS="-fPIC" \
       -DCUDA_PROPAGATE_HOST_FLAGS=OFF \
       -DTORCH_CUDA_ARCH_LIST=%{gpu_target_arch} \
%if "%{toolchain}" == "gcc"
%if %{have_cuda_gcc}
%if 0%{?fedora} > 34
       -DCUDA_HOST_COMPILER="%{_bindir}/cuda-g++" \
       -DCMAKE_CUDA_HOST_COMPILER="%{_bindir}/cuda-g++" \
%endif
%endif
%endif
%if "%{toolchain}" == "clang"
%if 0%{?fedora} > 40
       -DCUDA_HOST_COMPILER="%{_bindir}/clang++-18" \
       -DCMAKE_CUDA_HOST_COMPILER="%{_bindir}/clang++-18" \
%else
%if 0%{?fedora} > 39
       -DCUDA_HOST_COMPILER="%{_bindir}/clang++-17" \
       -DCMAKE_CUDA_HOST_COMPILER="%{_bindir}/clang++-17" \
%else
       -DCUDA_HOST_COMPILER="%{_bindir}/clang++-16" \
       -DCMAKE_CUDA_HOST_COMPILER="%{_bindir}/clang++-16" \
%endif
%endif
%endif
       -DCUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda-%{vcu_maj}.%{vcu_min}" \
       -DCMAKE_CUDA_COMPILER="/usr/local/cuda-%{vcu_maj}.%{vcu_min}/bin/nvcc" \
       -DCUDA_NVCC_FLAGS="--compiler-options;-fPIC;-Wno-deprecated-gpu-targets;-allow-unsupported-compiler;--fatbin-options;-compress-all" \
       -DCMAKE_CUDA_FLAGS="--compiler-options -fPIC -Wno-deprecated-gpu-targets -allow-unsupported-compiler --fatbin-options -compress-all" \
       -DNCCL_INCLUDE_DIR="%{_includedir}/nccl" \
%if %{use_magma}
       -DUSE_MAGMA=ON \
%else
       -DUSE_MAGMA=OFF \
%endif
       -DBUILD_SPLIT_CUDA=ON \
%if %{have_tensorrt}
       -DUSE_TENSORRT=ON \
%else
       -DUSE_TENSORRT=OFF \
%endif
%endif
       -DBLAS="OpenBLAS" \
       -DUSE_MPI=OFF \
       -DUSE_OBSERVERS=OFF \
       -DUSE_ASAN=OFF \
       -DUSE_ROCM=OFF \
%if %{use_dnnl}
       -DUSE_MKLDNN=ON \
%else
       -DUSE_MKLDNN=OFF \
%endif
%ifarch x86_64
       -DUSE_FBGEMM=ON \
%else
       -DUSE_FBGEMM=OFF \
%endif
       -DUSE_NNPACK=ON \
       -DUSE_QNNPACK=ON \
       -DUSE_PYTORCH_QNNPACK=ON \
       -DUSE_SYSTEM_FP16=ON \
       -DUSE_SYSTEM_PSIMD=ON \
%if %{ext_sleef}
       -DUSE_SYSTEM_SLEEF=ON \
%else
       -DUSE_SYSTEM_SLEEF=OFF \
%endif
       -DUSE_SYSTEM_FXDIV=ON \
%if %{ext_xnnpack}
       -DUSE_SYSTEM_XNNPACK=ON \
%else
       -DUSE_SYSTEM_XNNPACK=OFF \
%endif
       -DUSE_SYSTEM_CPUINFO=ON \
       -DUSE_SYSTEM_PTHREADPOOL=ON \
       -DUSE_TENSORPIPE=ON \
       -DUSE_FAKELOWP=OFF \
       -DUSE_OPENCL=OFF \
       -DUSE_GLOO=ON \
       -DUSE_LLVM=OFF \
       -DATEN_NO_TEST=ON

#make %{?_smp_mflags}
%define flash_attn_target flash_attention
stdbuf -oL ninja -v -j8 %{flash_attn_target}
stdbuf -oL ninja -v %{?_smp_mflags}
popd


%install

#
# install libraries
#

pushd build
export PYTHON_EXECUTABLE="%{__python3}"
export LD_LIBRARY_PATH="/usr/local/cuda-%{vcu_maj}.%{vcu_min}/%{_lib}/"
DESTDIR=%{buildroot} ninja install

# libraries
mkdir -p %{buildroot}%{_libdir}
find %{buildroot}/ -name "*.a" -type f -prune -exec rm -rf '{}' '+'
rm -rf %{buildroot}/usr/lib/python*
mv -f %{buildroot}/usr/lib/* %{buildroot}%{_libdir}/
popd
install -D -pm 755 build/lib/libnnapi_backend.so %{buildroot}/%{_libdir}/

# pyyhon wrappers
mkdir -p %{buildroot}/%{python3_sitearch}/torch/bin
install -D -pm 644 build/lib/_C.so %{buildroot}/%{python3_sitearch}/torch/
install -D -pm 644 aten/src/THC/THCDeviceUtils.cuh %{buildroot}/%{_includedir}/THC/

# symlinks
ln -sf %{_includedir} %{buildroot}/%{python3_sitearch}/torch/include
ln -sf %{_libdir} %{buildroot}/%{python3_sitearch}/torch/lib
ln -sf %{_bindir}/torch_shm_manager %{buildroot}/%{python3_sitearch}/torch/bin/torch_shm_manager

#
# install python bits
#

# torch
for f in `find ./torch/ -name '*.py'`;
do
  install -D -pm 644 $f %{buildroot}/%{python3_sitearch}/$f
done
# torchgen
for f in `find ./torchgen/ -name '*.py'`;
do
  install -D -pm 644 $f %{buildroot}/%{python3_sitearch}/$f
done
# functorch
for f in `find ./functorch/ -name '*.py'`;
do
  install -D -pm 644 $f %{buildroot}/%{python3_sitearch}/$f
done


%if 0%{?rhel} == 8
# py3.6 compat
sed -i 's|python_version = 3.7|python_version = 3.6|g' mypy*.ini
sed -i 's|python_min_version = (3, 7, 0)|python_min_version = (3, 6, 0)|' setup.py
find %{buildroot}/%{python3_sitearch} -name '*.py' -exec sed -i '/from __future__ import annotations/d' {} +
%endif

# version.py
cuver=$(/usr/local/cuda/bin/nvcc --version | grep release | cut -d',' -f2 | awk '{print $2}')
echo "from typing import Optional" > %{buildroot}/%{python3_sitearch}/torch/version.py
echo "__all__ = ['__version__', 'debug', 'cuda', 'git_version', 'hip']" >> %{buildroot}/%{python3_sitearch}/torch/version.py
echo "__version__ = '%{version}'" >> %{buildroot}/%{python3_sitearch}/torch/version.py
echo "debug = False" >> %{buildroot}/%{python3_sitearch}/torch/version.py
echo "cuda: Optional[str] = '$cuver'" >> %{buildroot}/%{python3_sitearch}/torch/version.py
echo "git_version = '%{schash0}'" >> %{buildroot}/%{python3_sitearch}/torch/version.py
echo "hip: Optional[str] = None" >> %{buildroot}/%{python3_sitearch}/torch/version.py

# install path
mv -f %{buildroot}/%{_builddir}/pytorch/nvfuser/nvfuser.so \
      %{buildroot}/%{_libdir}/ || true
mv -f %{buildroot}/%{_builddir}/pytorch/torch/lib/libnvfuser_codegen.so \
      %{buildroot}/%{_libdir}/ || true

# remove junk
rm -rf %{buildroot}/%{_includedir}/fmt || true
rm -rf %{buildroot}/%{_includedir}/clog.h || true
rm -rf %{buildroot}/%{_includedir}/xnnpack.h || true
rm -rf %{buildroot}/%{_builddir}/pytorch/test || true
rm -rf %{buildroot}/%{_builddir}/pytorch/nvfuser || true
rm -rf %{buildroot}/%{_libdir}/cmake/fmt || true
rm -rf %{buildroot}/%{_libdir}/cmake/ittapi || true
rm -rf %{buildroot}/%{_libdir}/cmake/KleidiAI || true
rm -rf %{buildroot}/%{_libdir}/cmake/sleef || true
rm -rf %{buildroot}/%{_libdir}/cmake/mimalloc* || true
rm -rf %{buildroot}/%{_libdir}/pkgconfig/mimalloc* || true
rm -rf %{buildroot}/%{_libdir}/pkgconfig/fmt.pc || true
rm -rf %{buildroot}/%{_libdir}/pkgconfig/sleef.pc || true

# egg info
%{python3} setup.py egg_info
cp -r torch.egg-info %{buildroot}%{python3_sitearch}/
sed -i '/^\[/!s/[<=>].*//g' %{buildroot}%{python3_sitearch}/*.egg-info/requires.txt
sed -i '/triton/d' %{buildroot}%{python3_sitearch}/*.egg-info/requires.txt
%if 0%{?rhel}
sed -i '/sympy/d' %{buildroot}%{python3_sitearch}/*.egg-info/requires.txt
sed -i '/fsspec/d' %{buildroot}%{python3_sitearch}/*.egg-info/requires.txt
%endif
sed -i 's|Requires-Dist: ||g' %{buildroot}%{python3_sitearch}/*.egg-info/PKG-INFO || true
sed -i 's|Provides-Extra: ||g' %{buildroot}%{python3_sitearch}/*.egg-info/PKG-INFO || true

# strip elf
set +x
find %{buildroot} -type f -print | LC_ALL=C sort |
  file -N -f - | sed -n -e 's/^\(.*\):[ \t]*.*ELF.*, not stripped.*/\1/p' |
  xargs --no-run-if-empty stat -c '%h %D_%i %n' |
  while read nlinks inum f; do
      echo "Stripping: $f"
      strip -s $f
  done
set -x


%files
%doc README.md
%doc CONTRIBUTING.md
%license LICENSE
%{_bindir}/*
%{_libdir}/libshm.so.*
%{_libdir}/libc10.so.*
%{_libdir}/libc10_cuda.so
%{_libdir}/libtorch.so.*
%{_libdir}/libtorch_cpu.so.*
%{_libdir}/libtorch_cuda.so
%{_libdir}/libtorch_global_deps.so.*
%{_libdir}/libcaffe2_nvrtc.so
%{_libdir}/libnnapi_backend.so
%{_libdir}/libtorch_cuda_linalg.so

%files devel
%{_includedir}/*
%{_datadir}/*
%{_libdir}/libshm.so
%{_libdir}/libc10.so
%{_libdir}/libtorch.so
%{_libdir}/libtorch_cpu.so
%{_libdir}/libtorch_global_deps.so

%files python3
%{python3_sitearch}/*
%{_libdir}/libtorch_python.so*


%changelog
* Sun Apr 07 2019 Balint Cristian <cristian.balint@gmail.com>
- github update releases