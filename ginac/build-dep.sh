#!/bin/bash

cd $(dirname $0)

mkdir build
cd build
tar -xvf ../cln-1.3.7.tar.bz2
cd cln-1.3.7

patch m4/intparam.m4 ../../patch/intparam.m4.patch
patch src/timing/cl_t_current2.cc ../../patch/hz.patch

autoreconf -ivf

export CFLAGS="-fexceptions"
export  CXXFLAGS="${CFLAGS} -std=c++20"
export  CPPFLAGS="${CFLAGS} -DNO_ASM"
export  LDFLAGS="${CFLAGS}"



emconfigure sh ./configure --host none --prefix $(pwd)/../install --without-gmp && make -j install



cd ../
tar -xvf ../ginac-1.8.9.tar.bz2
cd ginac-1.8.9


export CFLAGS="-fexceptions"
export CXXFLAGS="${CFLAGS} -std=c++20"
export CPPFLAGS="${CFLAGS}"
export LDFLAGS="${CFLAGS}"



emconfigure sh -c "./configure --host none --prefix $(pwd)/../install PKG_CONFIG_PATH=\"$(pwd)/../install/lib/pkgconfig\"" && make -j install

cd ../../