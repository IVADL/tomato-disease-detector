# PDD(Plant Diseases Detection) Prototype

- Dataset: https://www.aihub.or.kr/aidata/129
- Model
  - Detection
  - Yolov5 Github : https://github.com/ultralytics/yolov5
- Tool
  - Frontend : Streamlit - https://docs.streamlit.io/en/stable/#
  - Backend : FastAPI - https://fastapi.tiangolo.com/
  - Annotation Tool : CVAT - https://github.com/openvinotoolkit/cvat

## Env Setting

1. Clone food-image-classifier Repository

   ```sh
   $ git clone https://github.com/IVADL/PDD-prototype.git
   ```

2. docker-compose commands

   ```sh
   $ docker-compose build
   $ docker-compose up
   ```

3. Visit Streamlit UI

- visit http://localhost:8501 (http://localhost:8501)

4. Run model

- Select model : mobilenet or yolov5
- Test Image upload
- Click 'Detect Plant Disease Button'
