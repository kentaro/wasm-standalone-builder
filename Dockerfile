# syntax=docker/dockerfile:1.4
FROM rust:latest

WORKDIR /workspaces

ENV TVM_HOME /workspaces/tvm
ENV PYTHONPATH $TVM_HOME/python:${PYTHONPATH}

# Use a cache directory for cargo
# https://note.com/tkhm_dev/n/n439a4b4b9422
ENV CARGO_BUILD_TARGET_DIR=/tmp/target

# Build TVM and related libraries
# https://tvm.apache.org/docs/install/from_source.html

# Install dependencies
RUN <<EOS
  apt-get update
  apt-get install -y python3 python3-dev python3-pip python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev llvm libclang-dev
EOS

# Build TVM
RUN git clone --recursive https://github.com/apache/tvm tvm
RUN <<EOS
  cd tvm
  mkdir -p build
  cp cmake/config.cmake build/
  cd build
  sed -i 's/set(USE_LLVM OFF)/set(USE_LLVM ON)/g' config.cmake
  cmake ..
  make -j4
EOS

# Install python libraries
RUN <<EOS
  # for TVM
  pip3 install numpy decorator attrs
  # for building a ML model
  pip3 install onnx pillow psutil scipy
  # TVM python bindings
  cd tvm/python
  python3 setup.py install
EOS

# Build a ML model
# https://github.com/apache/tvm/tree/main/apps/wasm-standalone
RUN <<EOS
  cd tvm/apps/wasm-standalone/wasm-graph/tools
  LLVM_AR=llvm-ar-11 python3 ./build_graph_lib.py -O3
EOS

# Build a Wasm binary
COPY wasm-graph/ /workspaces/wasm-graph

RUN <<EOS
  cd wasm-graph

  # `libgraph_wasm32.a` is built in the previous step
  cp /workspaces/tvm/apps/wasm-standalone/wasm-graph/lib/libgraph_wasm32.a lib/

  rustup target add wasm32-unknown-unknown

  # `BINDGEN_EXTRA_CLANG_ARGS` is required to include clang headers
  # https://docs.rs/bindgen/latest/bindgen/struct.Builder.html#clang-arguments
  BINDGEN_EXTRA_CLANG_ARGS="-I/usr/lib/llvm-11/lib/clang/11.0.1/include" cargo build --release --target wasm32-unknown-unknown
EOS

RUN <<EOS
  rustup component add clippy
EOS
