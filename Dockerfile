FROM ubuntu:16.04
MAINTAINER Marchiano Oh <github mdjoh>

RUN apt-get update
RUN apt-get install -y python3-pip

RUN pip3 install --upgrade pip
RUN pip3 install numpy pandas


RUN mkdir -p /opt/
COPY ./example.py /opt/

# everything up to this point runs fine. python and libraries get installed correctly

# Run example.py when the container launches
# ENTRYPOINT ["python3", "example.py"]
