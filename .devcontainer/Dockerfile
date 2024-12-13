FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime

# qt5-default has bug so we need to install qtbase5-dev, etc.
# Install additional packages
RUN apt-get update && apt-get -y install --no-install-recommends \
libgl1-mesa-glx libgtk2.0-dev pkg-config \
libxcb-xinerama0-dev libgl1-mesa-dev \
libxcb*-dev libfontconfig1-dev libxkbcommon-x11-dev libgtk-3-dev \
qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools \
&& apt-get clean && rm -rf /var/lib/apt/lists/*

# Install python packages from requirements.txt
COPY ./requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt
