# Tomato Disease Detector
![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FIVADL/tomato-disease-detector) ![License](https://img.shields.io/github/license/IVADL/tomato-disease-detector?style=plastic) ![Stars](https://img.shields.io/github/stars/IVADL/PDD-prototype?style=social)

![love and jo](https://user-images.githubusercontent.com/38045080/115104832-01db6b80-9f96-11eb-9710-2f146eabe04f.png)

The repository is a Detector project that allows you to easily detect tomato's disease using Simple Web services. Currently, a total of 7 classes of disease can be detected with bounding box.

- [x] Image file available
- [x] Video file available
- [x] New yolo models can be added
- [ ] Other format models can be added

## Table of Contents

- [Structure](#Structure)
- [Usage](#Usage)
- [Examples](#Examples)
- [Team](#Team)
- [License](#License)

## Structure
- Dataset: https://www.aihub.or.kr/aidata/129
- Model
  - Detection
  - Yolov5 Github : https://github.com/ultralytics/yolov5
- Tool
  - Frontend : Streamlit - https://docs.streamlit.io/en/stable/#
  - Backend : FastAPI - https://fastapi.tiangolo.com/
  - Annotation Tool : CVAT - https://github.com/openvinotoolkit/cvat

## Usage

1. Clone This Repository

   ```sh
   $ git clone https://github.com/IVADL/PDD-prototype.git
   ```

2. docker-compose commands

   ```sh
   $ docker-compose build
   $ docker-compose up
   ```

3. Visit Streamlit UI

- visit [http://localhost:8501](http://localhost:8501)

4. Run model

- Select model : yolov5
- Test Image or video upload
- Click 'Detect Plant Disease Button'

## Example

1. Image Detect
![test112](https://user-images.githubusercontent.com/38045080/115104560-4960f800-9f94-11eb-9580-271a6650d50d.gif)
2. Video Detect
![test113](https://user-images.githubusercontent.com/38045080/115104630-beccc880-9f94-11eb-95af-87734ff3d2d8.gif)
![test114](https://user-images.githubusercontent.com/38045080/115104633-c9875d80-9f94-11eb-8573-af2a3ff95bd6.gif)

## Team

The project was conducted at the Korea Lab of Artificial Intelligence and formed a team called IVADL.

- Harim Kang [Git-hub](https://github.com/harim4422)
- Seonmin Kim [Git-hub](https://github.com/SeonminKim1)
- Eunbi Park [Git-hub](https://github.com/bluvory)

License
----

MIT
