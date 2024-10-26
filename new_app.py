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

# Suitable colors for each part
color_options = {
    'hair': ['#000000', '#6A4C93', '#FF6F61', '#F9D74D', '#A7C5EB'],  # Black, Purple, Coral, Yellow, Light Blue
    'lip': ['#EDBAD1', '#E3B9C1', '#FF6F61', '#E34D4D', '#D9A39A'],  # Pinkish shades
    'eyebrow': ['#3C2B1F', '#5D473A', '#6A4C93', '#000000', '#4F3A3C'],  # Brownish and Black
    'nose_face_neck': ['#F5C6A5', '#EED6D7', '#E3B9C1', '#DABBA5', '#F1E6D4'],  # Skin tones
}

# Color palette function
def color_palette(colors):
    cols = st.sidebar.columns(len(colors))
    selected_color = None
    for col, color in zip(cols, colors):
        if col.button('', key=color, style=f"background-color: {color}; width: 50px; height: 50px;"):
            selected_color = ImageColor.getcolor(color, "RGB")
    return selected_color

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

# Use color_palette function for each makeup part
hair_color = color_palette(color_options['hair']) or (0, 0, 0)  # Default to black if no color is selected
lip_color = color_palette(color_options['lip']) or (237, 189, 209)  # Default to a light pink
eyebrow_color = color_palette(color_options['eyebrow']) or (60, 43, 31)  # Default to brown
nose_face_neck_color = color_palette(color_options['nose_face_neck']) or (245, 198, 165)  # Default to a skin tone

# Saturation adjustments
hair_saturation = st.sidebar.slider("Hair Saturation", 0.5, 2.0, 1.0)
lip_saturation = st.sidebar.slider("Lip Saturation", 0.5, 2.0, 1.0)
eyebrow_saturation = st.sidebar.slider("Eyebrow Saturation", 0.5, 2.0, 1.0)
nose_face_neck_saturation = st.sidebar.slider("Nose, Face, and Neck Saturation", 0.5, 2.0, 1.0)

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
st.markdown('<footer style="text-align: center; margin-top: 20px; font-size: 14px; color: #555;">Created with ❤️ by [Your Name]</footer>', unsafe_allow_html=True)
