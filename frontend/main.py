import streamlit as st

import io
from PIL import Image
import requests
import tempfile
from requests_toolbelt.multipart.encoder import MultipartEncoder

backend = "http://fastapi:8000"


def detect_image(data, server):
    m = MultipartEncoder(fields={"file": ("filename", data, "image/jpeg")})

    resp = requests.post(
        server + "/detection/image",
        data=m,
        headers={"Content-Type": m.content_type},
        timeout=8000,
    )

    return resp


def detect_video(data, server):
    # fm = MultipartEncoder(fields={"file": ("filename", data, "text")})
    m = MultipartEncoder(fields={"file": ("filename", data, "video/mp4")})

    resp = requests.post(
        server + "/detection/video",
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

    data_type = st.selectbox("Choose Data Type", ["Image", "Video"])
    input_data = st.file_uploader(f"insert {data_type}")  # image upload widget

    if st.button("Detect Plant Disease"):

        col1, col2 = st.beta_columns(2)

        if data_type == "Image":

            if input_data:
                pred = detect_image(input_data, backend)
                original_image = Image.open(input_data).convert("RGB")
                converted_image = pred.content
                converted_image = Image.open(io.BytesIO(converted_image)).convert("RGB")
                r, g, b = converted_image.split()
                converted_image = Image.merge("RGB", (b,g,r))

                col1.header("Original")
                col1.image(original_image, use_column_width=True)
                col2.header("Detected")
                col2.image(converted_image, use_column_width=True)

            else:
                # handle case with no image
                st.write("Insert an image!")

        elif data_type == "Video":
            col1.header("Original")
            col2.header("Detected")
            col1.video(input_data.read(), format="video/mp4")
            # video_path = "/var/lib/assets/video1.mp4"
            # with open(video_path, "wb") as wfile:
            #     wfile.write(input_data.read())
            resp = detect_video(input_data.read(), backend)
            detected_content = resp.content
            detected_content = io.BytesIO(detected_content)
            col2.video(detected_content, format="video/mp4")
            # col2.video(pred, format="video/mp4")

if __name__ == "__main__":
    main()