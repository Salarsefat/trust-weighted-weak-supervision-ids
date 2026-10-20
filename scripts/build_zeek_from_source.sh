#!/usr/bin/env bash
# The Zeek build that was used for validation (Zeek 8.0.10, Ubuntu 24.04).
# Binary packages are an alternative: https://docs.zeek.org/en/lts/install.html
set -euo pipefail
PREFIX="${PREFIX:-/opt/zeek}"
VERSION="${VERSION:-v8.0.10}"
sudo apt-get update
sudo apt-get install -y cmake ninja-build make gcc g++ flex bison libfl-dev libpcap-dev libssl-dev zlib1g-dev \
                        python3 python3-dev swig ccache git
git clone --recursive --depth 1 --branch "$VERSION" https://github.com/zeek/zeek zeek-src
cd zeek-src
./configure --prefix="$PREFIX" --generator=Ninja --build-type=Release --ccache \
  --disable-spicy --disable-zeekctl --disable-auxtools --disable-btest --disable-broker-tests \
  --disable-python --disable-javascript --disable-zkg --disable-cpp-tests --disable-zeek-client \
  --disable-cluster-backend-zeromq --disable-af-packet
cmake --build build -j"$(nproc)"
sudo cmake --install build
"$PREFIX/bin/zeek" --version
