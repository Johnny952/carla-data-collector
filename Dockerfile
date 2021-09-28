FROM python:3.7

RUN apt-get update
RUN apt-get install -y gnupg2
RUN echo "deb http://us.archive.ubuntu.com/ubuntu/ bionic main" >> /etc/apt/sources.list
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 3B4FE6ACC0B21F32
RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install libjpeg-turbo8 -y

COPY requirements.txt requirements.txt
COPY gym_carla-0.1.0-py3-none-any.whl gym_carla-0.1.0-py3-none-any.whl
COPY lbc_agent-0.1.0-py3-none-any.whl lbc_agent-0.1.0-py3-none-any.whl
RUN pip install -r requirements.txt
RUN pip install gym_carla-0.1.0-py3-none-any.whl lbc_agent-0.1.0-py3-none-any.whl

WORKDIR /home/carla_collector
COPY . .
