FROM ubuntu
RUN cd root
WORKDIR /root
ADD model_prediction.py /root
ADD requirements.txt /root
ADD test_model_prediction.py /root

RUN apt-get update
RUN apt-get -y install python3
RUN apt-get install -y python3-pip
RUN pip3 install -r requirements.txt

ENV PORT 3211

EXPOSE 3211

ENTRYPOINT ["python3", "model_prediction.py"]