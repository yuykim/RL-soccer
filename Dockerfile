FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get --no-install-recommends install -yq git cmake build-essential \
  libgl1-mesa-dev libsdl2-dev \
  libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
  libdirectfb-dev libst-dev mesa-utils xvfb x11vnc \
  ffmpeg wget tree \
  python3-pip

RUN python3 -m pip install --upgrade pip setuptools wheel
RUN python3 -m pip install psutil six pillow stable-baselines3
RUN python3 -m pip install torch torchvision

COPY gfootball /gfootball
RUN cd /gfootball && python3 -m pip install .
WORKDIR '/gfootball'
CMD ["/bin/bash"]
