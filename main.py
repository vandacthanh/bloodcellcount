import streamlit as st
import requests
import base64
import io
from PIL import Image
import glob
from base64 import decodebytes
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import json

st.set_page_config(layout="wide", page_icon="💻", page_title="ỨNG DỤNG NHẬN DẠNG TẾ BÁO MÁU")
# Set up basic information
c1, c2, c3 = st.columns(3)
c1.image("images/ICON_1.jpg")
c2.image("images/logo.png")
c3.image("images/streamlit_logo.png")
st.markdown("<h2 style='text-align: center; color: green;'; font-family:'Courier New'>ỨNG DỤNG NHẬN DẠNG TẾ BÁO MÁU</h2>", unsafe_allow_html=True)

# Input and parameters
st.markdown("<h3 style='text-align: left; color: green;'; font-family:'Courier New'>CÁC BƯỚC PHÂN TÍCH HÌNH ẢNH</h3>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: left; color: green;'; font-family:'Courier New'>BƯỚC 1: CHỌN ẢNH CẦN PHÂN TÍCH</h4>", unsafe_allow_html=True)

uploaded_file = st.file_uploader('', type=['png', 'jpg', 'jpeg'], accept_multiple_files=False)
st.write('[CHỌN ẢNH KHÁC TỪ KHO DỮ LIỆU](https://public.roboflow.com/object-detection/bccd/)')

st.markdown("<h4 style='text-align: left; color: green;'; font-family:'Courier New'>BƯỚC 2: HIỆU CHỈNH CÁC THÔNG SỐ</h4>", unsafe_allow_html=True)
C1, C2 = st.columns(2)
confidence_threshold = C1.slider('Ngưỡng tin cậy: chọn độ tin cậy thấp nhất để thể hiện biên của hộp nhận dạng:', 0.0, 1.0, 0.5, 0.01)
overlap_threshold = C2.slider('Ngưỡng giao ảnh: giá trị thấp nhát cho các khoảng giao nhau giữa các hộp nhận dạng:', 0.0, 1.0, 0.5, 0.01)

st.markdown("<h4 style='text-align: left; color: green;'; font-family:'Courier New'>BƯỚC 3: ĐÁNH GIÁ KẾT QUẢ VÀ LƯU TRỮ</h4>", unsafe_allow_html=True)
url = 'https://github.com/matthewbrems/streamlit-bccd/blob/master/BCCD_sample_images/BloodImage_00038_jpg.rf.6551ec67098bc650dd650def4e8a8e98.jpg?raw=true'
if uploaded_file: image = Image.open(uploaded_file)  
else: image = Image.open(requests.get(url, stream=True).raw)

# Convert to JPEG Buffer.
buffered = io.BytesIO()
image.save(buffered, quality=90, format='JPEG')

# Base 64 encode.
img_str = base64.b64encode(buffered.getvalue())
img_str = img_str.decode('ascii')

## Construct the URL to retrieve image.
upload_url = ''.join([
    'https://infer.roboflow.com/rf-bccd-bkpj9--1',
    '?access_token=vbIBKNgIXqAQ',
    '&format=image',
    f'&overlap={overlap_threshold * 100}',
    f'&confidence={confidence_threshold * 100}',
    '&stroke=2',
    '&labels=True'
])

## POST to the API.
r = requests.post(upload_url,
                  data=img_str,
                  headers={
    'Content-Type': 'application/x-www-form-urlencoded'
})

image = Image.open(BytesIO(r.content))

# Convert to JPEG Buffer.
buffered = io.BytesIO()
image.save(buffered, quality=90, format='JPEG')
_, C, _ = st.columns(3)
C.image(image, use_column_width=True)
## Construct the URL to retrieve JSON.
upload_url = ''.join([ 'https://infer.roboflow.com/rf-bccd-bkpj9--1', '?access_token=vbIBKNgIXqAQ'])

## POST to the API.
r = requests.post(upload_url,
                  data=img_str,
                  headers={
    'Content-Type': 'application/x-www-form-urlencoded'
})

## Save the JSON.
output_dict = r.json()

## Generate list of confidences.
confidences = [box['confidence'] for box in output_dict['predictions']]

## Summary statistics section in main app.
st.markdown("<h4 style='text-align: left; color: green;'; font-family:'Courier New'>BƯỚC 4: KẾT QUẢ THỐNG KÊ</h4>", unsafe_allow_html=True)
C1, C2 = st.columns(2)
C1.markdown(f"<h5 style='text-align: center; color: green;'; font-family:'Courier New'> - SỐ LƯỢNG TẾ BÀO TRONG ẢNH: {len(confidences)} </h5>", unsafe_allow_html=True)
C2.markdown(f"<h5 style='text-align: center; color: green;'; font-family:'Courier New'> - ĐỘ TIN CẬY TRUNG BÌNH: {(np.round(np.mean(confidences),4))}</h5>", unsafe_allow_html=True)

# Cell count
output_string = json.dumps(output_dict)
platelets_num = output_string.count('Platelets')
wbc_num = output_string.count('WBC')
rbc_num = output_string.count('RBC')
c1, c2, c3 = st.columns(3)
c1.markdown(f"<h5 style='text-align: center; color: grey;'; font-family:'Courier New'> - SỐ LƯỢNG TIỂU CẦU TRONG ẢNH: {platelets_num} </h5>", unsafe_allow_html=True)
c2.markdown(f"<h5 style='text-align: center; color: purple;'; font-family:'Courier New'> - SỐ LƯỢNG BẠCH CẦU TRONG ẢNH: {wbc_num} </h5>", unsafe_allow_html=True)
c3.markdown(f"<h5 style='text-align: center; color: red;'; font-family:'Courier New'> - SỐ LƯỢNG HỒNG CẦU TRONG ẢNH: {rbc_num} </h5>", unsafe_allow_html=True)

## Histogram in main app.
_, C, _ = st.columns(3)
fig, ax = plt.subplots()
ax.hist(confidences, bins=10, range=(0.0,1.0))
C.pyplot(fig)
C.markdown("<h4 style='text-align: center; color: green;'; font-family:'Courier New'>BIỂU ĐỒ HISTOGRAM VỀ ĐỘ TIN CẬY</h4>", unsafe_allow_html=True)

# Footer
footer = """
<html>
<head>
<style>

footer {
text-align: center;
padding: 10px;
background-color: purple;
color: white;
}
</style>
</head>
<body>

<footer>
<p> THIẾT KẾ VÀ LẬP TRÌNH: NGHIÊN CỨU VIÊN VĂN ĐẮC THÀNH<br>
<p> PHÒNG THÍ NGHIỆM MỞ <br>
<p> © 2022 VIỆN KHOA HỌC VÀ CÔNG NGHỆ TÍNH TOÁN TPHCM. All rights reserved.<br>
<a href="email: thanh.vd@icst.org.vn">thanh.vd@icst.org.vn</a></p>
</footer>

</body>
</html>
"""
st.markdown(footer, unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: green;'; font-family:'Courier New'></h4>", unsafe_allow_html=True)
