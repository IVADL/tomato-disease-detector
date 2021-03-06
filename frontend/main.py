import streamlit as st

import io
from PIL import Image
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder

backend = "http://fastapi:8000"


def detect(image, server):
    m = MultipartEncoder(fields={"file": ("filename", image, "image/jpeg")})

    resp = requests.post(
        server + "/detection",
        data=m,
        headers={"Content-Type": m.content_type},
        timeout=8000,
    )

    return resp


def main():
    st.title("Plant Disease Detector Application")

    st.write(
        """Test Plant Image
            This streamlit example uses a FastAPI service as backend.
            Visit this URL at `:8000/docs` for FastAPI documentation."""
    )  # description and instructions

    # Side Bar
    st.sidebar.title("Test Models")
    app_mode = st.sidebar.selectbox("Choose Model", ["YOLO_V5_202103"])

    if app_mode == "YOLO_V5_202103":
        run_app()


def run_app():

    input_image = st.file_uploader("insert image")  # image upload widget

    if st.button("Detect Plant Disease"):

        col1, col2 = st.beta_columns(2)

        if input_image:
            pred = detect(input_image, backend)
            original_image = Image.open(input_image).convert("RGB")
            converted_image = pred.content
            converted_image = Image.open(io.BytesIO(converted_image)).convert("RGB")
            col1.header("Original")
            col1.image(original_image, use_column_width=True)
            col2.header("Detected")
            col2.image(converted_image, use_column_width=True)

        else:
            # handle case with no image
            st.write("Insert an image!")


if __name__ == "__main__":
    main()