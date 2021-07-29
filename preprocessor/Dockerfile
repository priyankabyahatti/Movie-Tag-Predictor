FROM ubuntu
RUN cd root
WORKDIR /root
RUN mkdir .kaggle
ADD kaggle.json /root/.kaggle
ADD data_preprocess.py /root
ADD requirements.txt /root
ADD test_data_preprocess.py /root
COPY ./corpora /root/nltk_data/corpora

RUN chmod 600 /root/.kaggle/kaggle.json
RUN apt-get update
RUN apt-get -y install python3
RUN apt-get install -y python3-pip
RUN pip3 install -r requirements.txt
# RUN python3 -m nltk.downloader punkt
# RUN python3 -m nltk.downloader wordnet



ENV PORT 3111

EXPOSE 3111

ENTRYPOINT ["python3", "data_preprocess.py"]