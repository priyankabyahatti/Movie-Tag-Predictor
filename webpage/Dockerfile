FROM ubuntu

WORKDIR /app

COPY . .

RUN apt-get update
RUN apt-get -y install python3
RUN apt-get install -y python3-pip
RUN pip3 install -r requirements.txt

ENV PORT 3311

EXPOSE 3311

ENTRYPOINT ["python3", "main.py"]