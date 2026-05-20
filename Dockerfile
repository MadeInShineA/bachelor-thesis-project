# Adapted from
# https://github.com/glatard/fmriprep-1/blob/master/Dockerfile_fuzzy
# https://github.com/verificarlo/verificarlo/blob/master/doc/01-Install.md
# https://github.com/verificarlo/fuzzy#using-fuzzy-in-multi-stage-builds

# Qwen3.6-Plus was used for troubleshooting (mostly regarding make issues)

FROM verificarlo/fuzzy:v0.9.1-lapack-python3.8.5-numpy-scipy-sklearn AS fuzzy

FROM nipreps/fmriprep:25.2.5

# Install Verificarlo dependencies
RUN apt-get -y update && apt-get install libmpfr-dev clang llvm-dev parallel bzip2 gcc autoconf automake libtool build-essential python3 python3-pip -y

RUN git clone https://github.com/verificarlo/verificarlo.git

RUN cd verificarlo && git checkout v2.4.0

RUN cd verificarlo && ./autogen.sh

ENV PYTHON=/usr/bin/python3

ENV PYTHON3=/usr/bin/python3

RUN cd verificarlo && \
    ./configure --without-flang --with-llvm=$(llvm-config --prefix) --without-prism

RUN cd verificarlo && make install-interflop-stdlib

RUN cd verificarlo && make

RUN cd verificarlo && make install

# Copy libmath fuzzy environment from fuzzy image, for example
RUN mkdir -p /opt/mca-libmath/{fast,standard,quad,mpfr}
COPY --from=0 /opt/mca-libmath/set-fuzzy-libmath.py /usr/local/bin/set-fuzzy-libmath
COPY --from=0 /opt/mca-libmath/fast/libmath.so /opt/mca-libmath/fast/libmath.so
COPY --from=0 /opt/mca-libmath/standard/libmath.so /opt/mca-libmath/standard/libmath.so
COPY --from=0 /opt/mca-libmath/quad/libmath.so /opt/mca-libmath/quad/libmath.so
COPY --from=0 /opt/mca-libmath/mpfr/libmath.so /opt/mca-libmath/mpfr/libmath.so
COPY --from=0 /usr/local/lib/libinterflop* /usr/local/lib/

# If you will also want to recompile more libraries with verificarlo, add these lines
COPY --from=fuzzy /usr/local/bin/verificarlo* /usr/local/bin/
COPY --from=fuzzy /usr/local/include/* /usr/local/include/

# Preloading the instrumented shared library
ARG FUZZY_LIBMATH_VERSION=standard
RUN set-fuzzy-libmath --version=${FUZZY_LIBMATH_VERSION}

ENV VFC_BACKENDS='libinterflop_mca.so --precision-binary32=24 --precision-binary64=53 --mode=mca'

ENV LD_PRELOAD=/opt/mca-libmath/${FUZZY_LIBMATH_VERSION}/libmath.so

ENV VFC_LOGGING=1
