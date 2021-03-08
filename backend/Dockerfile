FROM tiangolo/uvicorn-gunicorn:python3.8-slim

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install git -y

RUN python -m pip install --upgrade pip

RUN mkdir /backend

COPY requirements.txt /backend

WORKDIR /backend

RUN pip install -r requirements.txt

COPY . /backend

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]