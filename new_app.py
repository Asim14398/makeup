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

# Custom CSS for gradient title, sidebar, and footer styling
st.markdown("""
    <style>
        .app-title {
            font-size: 3em;
            text-align: center;
            color: #fff;
            background: linear-gradient(90deg, #FF7F50, #FF4500, #FFD700);
            padding: 0.5em;
            border-radius: 10px;
            margin-bottom: 1em;
        }
        .sidebar .sidebar-content {
            background-color: #FAEBD7;
            padding: 1em;
            border-radius: 10px;
        }
        .sidebar .stTitle {
            color: #FF6347;
        }
        footer {
            text-align: center;
            margin-top: 20px;
            font-size: 14px;
            color: #555;
        }
    </style>
""", unsafe_allow_html=True)

# App title and layout styling
st.set_page_config(page_title='Virtual Makeup App', layout='wide')
st.markdown('<div class="app-title">🌟 Virtual Makeup App 🌟</div>', unsafe_allow_html=True)
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
    'blush': [5, 6],  # Assuming left and right cheeks
    'foundation': [1, 10, 14]  # Whole face and neck
}

# Suitable colors for each part (using color names)
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
        ('Espresso', '#3B2A24') 
    ],
    'blush': [
        ('Soft Pink', '#FFC0CB'), ('Peach', '#FFE5B4'), ('Rosy Red', '#FF6347'),
        ('Mauve', '#D8BFD8'), ('Deep Rose', '#C71585')
    ],
    'foundation': [
        ('Porcelain', '#FBE7D8'), ('Fair', '#F0D6C6'), ('Light Beige', '#E2B5A8'),
        ('Medium Beige', '#D1A57C'), ('Olive', '#BDA56D'), ('Tan', '#C89D68'),
        ('Caramel', '#AA7D4D')
    ]
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
blush_color_name, blush_color_hex = st.sidebar.selectbox("Select Blush Color", color_options['blush'], format_func=lambda x: x[0])
foundation_color_name, foundation_color_hex = st.sidebar.selectbox("Select Foundation Color", color_options['foundation'], format_func=lambda x: x[0])

# Saturation adjustments
hair_saturation = st.sidebar.slider("Hair Saturation", 0.5, 2.0, 1.0)
lip_saturation = st.sidebar.slider("Lip Saturation", 0.5, 2.0, 1.0)
eyebrow_saturation = st.sidebar.slider("Eyebrow Saturation", 0.5, 2.0, 1.0)
nose_face_neck_saturation = st.sidebar.slider("Nose, Face, and Neck Saturation", 0.5, 2.0, 1.0)
blush_saturation = st.sidebar.slider("Blush Saturation", 0.5, 2.0, 1.0)
foundation_saturation = st.sidebar.slider("Foundation Saturation", 0.5, 2.0, 1.0)

# Convert color codes to RGB format
hair_color = ImageColor.getcolor(hair_color_hex, "RGB")
lip_color = ImageColor.getcolor(lip_color_hex, "RGB")
eyebrow_color = ImageColor.getcolor(eyebrow_color_hex, "RGB")
nose_face_neck_color = ImageColor.getcolor(nose_face_neck_color_hex, "RGB")
blush_color = ImageColor.getcolor(blush_color_hex, "RGB")
foundation_color = ImageColor.getcolor(foundation_color_hex, "RGB")

# Colors and saturation values
colors = [hair_color, lip_color, lip_color, eyebrow_color, eyebrow_color, nose_face_neck_color, blush_color, foundation_color]
saturation = [hair_saturation, lip_saturation, lip_saturation, eyebrow_saturation, eyebrow_saturation, nose_face_neck_saturation, blush_saturation, foundation_saturation]

# Apply the colors for each part
for part, color, saturate in zip(table.values(), colors, saturation):
    if isinstance(part, list):
        for p in part:
            image = apply_clothing(image, parsing, p, color, saturate)
    else:
        image = apply_clothing(image, parsing, part, color, saturate)

# Display the final image
st.image(image, caption="Styled Image with Blush and Foundation", use_column_width=True)
