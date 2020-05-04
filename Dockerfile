FROM python:3.6
MAINTAINER Andre Lamurias (alamurias@lasige.di.fc.ul.pt)
WORKDIR /

RUN apt-get update -y && apt-get upgrade -y && apt-get install less vim wget git -y
#RUN apt-get update && apt-get install -y python3 python3-pip python3-dev && apt-get autoclean -y
RUN pip3 install --upgrade pip
RUN alias python=python3
RUN alias pip=pip3
RUN apt-get update && apt-get install -y default-jre && apt-get autoclean -y
#RUN pip3 install torch torchvision
COPY ./requirements.txt ./
RUN pip3 install -r requirements.txt --ignore-installed
RUN python -m spacy download en_core_web_lg
RUN python -m spacy download en
