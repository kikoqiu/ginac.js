#!/bin/bash
base_dir=$(dirname $0)

mkdir -p build/bind
cd build/bind

export PREFIX="$(pwd)/../install"
export CFLAGS="-I$PREFIX/include -fexceptions"
export CXXFLAGS="${CFLAGS} -std=c++20"
export LDFLAGS="-L$PREFIX/lib ${CFLAGS}"

emcc ${CXXFLAGS} -c ../../src/main.cpp -o main.o
emcc ${CXXFLAGS} -c ../../src/trigsimp.cpp -o trigsimp.o
emcc ${CXXFLAGS} -c ../../src/integ_ex.cpp -o integ_ex.o

emcc ${LDFLAGS} \
    -s ENVIRONMENT='web,worker,node' \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s MODULARIZE=1 \
    -s EXPORT_NAME="createGinacModule" \
    -s "EXPORTED_RUNTIME_METHODS=['HEAP8','HEAPU8','HEAP16','HEAPU16','HEAP32','HEAPU32','HEAPF32','HEAPF64','UTF8ToString','stringToUTF8','lengthBytesUTF8','getValue','setValue']" \
    -s EXPORTED_FUNCTIONS="['_malloc','_free']" \
    -s WARN_UNALIGNED=1 \
    -s ERROR_ON_UNDEFINED_SYMBOLS=0 \
    -s FILESYSTEM=1 \
    -s ASSERTIONS=0 \
    --emit-symbol-map \
    -lcln -lginac \
    main.o trigsimp.o integ_ex.o\
    -o ginac.js




cd $base_dir
