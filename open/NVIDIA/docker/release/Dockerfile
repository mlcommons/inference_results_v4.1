ARG BASE_IMAGE
FROM ${BASE_IMAGE} as base

ARG ENV
ARG BUILD_CONTEXT

ENV ENV=${ENV}
ENV BUILD_CONTEXT=${BUILD_CONTEXT}

SHELL ["/bin/bash", "-c"]

WORKDIR /tmp

COPY --from=common /requirements.${BUILD_CONTEXT}.* .

# 1. apt installs
COPY --from=common /install_apt_deps.sh .
RUN if [ -f install_apt_deps.sh ]; then bash install_apt_deps.sh; fi

# 2. pip installs
COPY --from=common /install_pip_deps.sh .
RUN if [ -f install_pip_deps.sh ]; then bash install_pip_deps.sh; fi

# 3. Any pip install or from-source build required.
COPY --from=common /install_misc_deps.sh .
RUN if [ -f install_misc_deps.sh ]; then bash install_misc_deps.sh; fi

# 4. switch between compiler for building faster transformer repo
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 9
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 11

WORKDIR /work
