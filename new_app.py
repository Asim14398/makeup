import cv2
import numpy as np
from skimage.filters import gaussian
from test import evaluate
import streamlit as st
from PIL import Image, ImageColor

# Sharpening function
def sharpen(img):
    img = img * 1.0
    gauss_out = gaussian(img, sigma=5, channel_axis=True)
    alpha = 1.5
    img_out = (img - gauss_out) * alpha + img
    img_out = np.clip(img_out / 255.0, 0, 1) * 255
    return np.array(img_out, dtype=np.uint8)

# Saturation adjustment function
def adjust_saturation(image, saturation_factor):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_hsv[:, :, 1] = np.clip(image_hsv[:, :, 1] * saturation_factor, 0, 255)
    return cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

# Makeup/clothing application function
def apply_clothing(image, parsing, part, color, saturation=1.0):
    r, g, b = color
    tar_color = np.zeros_like(image)
    tar_color[:, :, 0] = r
    tar_color[:, :, 1] = g
    tar_color[:, :, 2] = b

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)
    image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]
    changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    if part == 17:  # Assuming hair is part 17
        changed = sharpen(changed)

    changed[parsing != part] = image[parsing != part]

    if saturation != 1.0:
        changed = adjust_saturation(changed, saturation)

    return changed

# File path to demo image
DEMO_IMAGE = 'image/116.jpg'

# App title and layout styling
st.set_page_config(page_title='Virtual Makeup App', layout='wide')
st.title('🌟 Virtual Makeup App 🌟')
st.sidebar.title('🎨 Style Parameters')
st.sidebar.subheader('Choose your colors and adjust the saturation.')

# Adding parts for face
table = {
    'hair': 17,
    'upper_lip': 12,
    'lower_lip': 13,
    'left_eyebrow': 2,
    'right_eyebrow': 3,
    'nose_face_neck': [10, 1, 14],  # Grouped parts for face
}

# Expanded color options for various parts
color_options = {
    'hair': [
        ('Black', '#000000'), ('Dark Purple', '#6A4C93'), ('Coral', '#FF6F61'),
        ('Yellow', '#F9D74D'), ('Light Blue', '#A7C5EB'), ('Brown', '#8B4513'),
        ('Chocolate', '#D2691E'), ('Orange', '#FF4500'), ('Medium Violet Red', '#C71585'),
        ('Gold', '#FFD700')
    ],
    'lip': [
        ('Light Pink', '#EDBAD1'), ('Rosy Pink', '#E3B9C1'), ('Coral Red', '#FF6F61'),
        ('Red', '#E34D4D'), ('Tan', '#D9A39A'), ('Pale Pink', '#FFB6C1'),
        ('Hot Pink', '#FF69B4'), ('Pale Violet Red', '#DB7093'), ('Firebrick', '#B22222'),
        ('Indian Red', '#CD5C5C')
    ],
    'eyebrow': [
        ('Dark Brown', '#3C2B1F'), ('Medium Brown', '#5D473A'), ('Purple Brown', '#6A4C93'),
        ('Black', '#000000'), ('Dark Gray', '#4F3A3C'), ('Saddle Brown', '#8B4513'),
        ('Brown', '#A52A2A'), ('Chocolate', '#D2691E'), ('Dark Slate Gray', '#2F4F4F')
    ],
    'nose_face_neck': [
      ('Porcelain', '#FBE7D8'), ('Fair', '#F0D6C6'), ('Light Beige', '#E2B5A8'),
        ('Medium Beige', '#D1A57C'), ('Olive', '#BDA56D'), ('Tan', '#C89D68'),
        ('Caramel', '#AA7D4D'), ('Deep Tan', '#7A4B35'), ('Mocha', '#4E3B31'),
        ('Espresso', '#3B2A24'), ('Ebony', '#2C1B15')
    ],
}

# Image upload option
img_file_buffer = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", 'png'])

if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))
    demo_image = img_file_buffer
else:
    demo_image = DEMO_IMAGE
    image = np.array(Image.open(demo_image))

# Resize and display original image
h, w, _ = image.shape
image = cv2.resize(image, (1024, 1024))

# Run body parsing
cp = 'cp/79999_iter.pth'
parsing = evaluate(demo_image, cp)
parsing = cv2.resize(parsing, image.shape[0:2], interpolation=cv2.INTER_NEAREST)

# Color picker widgets for each part
hair_color_name, hair_color_hex = st.sidebar.selectbox("Select Hair Color", color_options['hair'], format_func=lambda x: x[0])
lip_color_name, lip_color_hex = st.sidebar.selectbox("Select Lip Color", color_options['lip'], format_func=lambda x: x[0])
eyebrow_color_name, eyebrow_color_hex = st.sidebar.selectbox("Select Eyebrow Color", color_options['eyebrow'], format_func=lambda x: x[0])
nose_face_neck_color_name, nose_face_neck_color_hex = st.sidebar.selectbox("Select Nose, Face, and Neck Color", color_options['nose_face_neck'], format_func=lambda x: x[0])

# Saturation adjustments
hair_saturation = st.sidebar.slider("Hair Saturation", 0.5, 2.0, 1.0)
lip_saturation = st.sidebar.slider("Lip Saturation", 0.5, 2.0, 1.0)
eyebrow_saturation = st.sidebar.slider("Eyebrow Saturation", 0.5, 2.0, 1.0)
nose_face_neck_saturation = st.sidebar.slider("Nose, Face, and Neck Saturation", 0.5, 2.0, 1.0)

# Convert color codes to RGB format
hair_color = ImageColor.getcolor(hair_color_hex, "RGB")
lip_color = ImageColor.getcolor(lip_color_hex, "RGB")
eyebrow_color = ImageColor.getcolor(eyebrow_color_hex, "RGB")
nose_face_neck_color = ImageColor.getcolor(nose_face_neck_color_hex, "RGB")

# Assigning colors and saturation to parts
colors = [
    hair_color, lip_color, lip_color, eyebrow_color,
    eyebrow_color, nose_face_neck_color
]

saturation = [
    hair_saturation, lip_saturation, lip_saturation, eyebrow_saturation,
    eyebrow_saturation, nose_face_neck_saturation
]

# Applying makeup changes
for part, color, saturate in zip(table.values(), colors, saturation):
    if isinstance(part, list):
        for p in part:
            image = apply_clothing(image, parsing, p, color, saturate)
    else:
        image = apply_clothing(image, parsing, part, color, saturate)

# Resize images back to original size
styled_image = cv2.resize(image, (w, h))
original_image = cv2.resize(np.array(Image.open(demo_image)), (w, h))

# Output images side by side
st.subheader('✨ Styled Output Image ✨')
col1, col2 = st.columns(2)
with col1:
    st.subheader('🖼️ Original Image')
    st.image(original_image, use_column_width=True)
with col2:
    st.subheader('🖼️ Styled Image')
    st.image(styled_image, use_column_width=True)

# Footer
st.markdown('<footer style="text-align: center; margin-top: 20px; font-size: 14px; color: #555;">Created with ❤️ by GUJJAR</footer>', unsafe_allow_html=True)
