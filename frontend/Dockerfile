FROM python:3.8-slim

RUN mkdir /frontend

COPY requirements.txt /frontend

WORKDIR /frontend

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y ffmpeg

RUN pip install -r requirements.txt

COPY . /frontend

EXPOSE 8501

CMD ["streamlit", "run", "main.py"]