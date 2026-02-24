import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
from skimage import transform as skimage_transform
from u2net_model import U2NET

# ---- Page config ----
st.set_page_config(
    page_title="AI Portrait Generator",
    page_icon="🎨",
    layout="centered"
)

# ---- Load model (cached so it only loads once) ----
@st.cache_resource
def load_model():
    net = U2NET(3, 1)
    net.load_state_dict(torch.load('u2net_portrait.pth', map_location='cpu'))
    net.eval()
    return net

# ---- Helper: normalize prediction ----
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    return (d - mi) / (ma - mi)

# ---- Helper: preprocess image for model ----
def preprocess(image):
    img = np.array(image.convert('RGB'))
    img = skimage_transform.resize(img, (512, 512), mode='constant')
    img = img / np.max(img)
    tmp = np.zeros((512, 512, 3))
    tmp[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
    tmp[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
    tmp[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225
    tmp = tmp.transpose((2, 0, 1))
    tensor = torch.from_numpy(tmp).unsqueeze(0).float()
    return tensor

# ---- Style A: AI Draw ----
def ai_draw(orig_np, d):
    pred  = 1.0 - normPRED(d[:, 0, :, :])
    pred  = normPRED(pred)
    mask  = pred.squeeze().cpu().data.numpy()
    portrait = Image.fromarray((mask * 255)).convert('RGB')
    return portrait

# ---- Style B: Pencil Sketch ----
def pencil_sketch(orig_np, d):
    # Grayscale
    gray     = cv2.cvtColor(orig_np, cv2.COLOR_RGB2GRAY)

    # Sharpen before sketching
    kernel   = np.array([[0, -1, 0], [-1, 9, -1], [0, -1, 0]])
    gray     = cv2.filter2D(gray, -1, kernel)

    # Dodge blend
    gray_inv  = cv2.bitwise_not(gray)
    gray_blur = cv2.GaussianBlur(gray_inv, (35, 35), 0)
    sketch    = cv2.divide(gray, cv2.bitwise_not(gray_blur), scale=256.0)

    # Darken lines
    sketch = np.clip(cv2.multiply(sketch, 0.75), 0, 255).astype(np.uint8)

    # Adaptive threshold for crisp edges
    edges  = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, blockSize=9, C=4
    )
    sketch = cv2.multiply(sketch, edges, scale=1/255.0).astype(np.uint8)

    # Saliency mask — clean white background
    pred       = normPRED(d[:, 0, :, :])
    mask_np    = pred.squeeze().cpu().data.numpy()
    mask       = cv2.resize((mask_np * 255).astype(np.uint8), (512, 512))
    mask       = cv2.GaussianBlur(mask, (21, 21), 0)
    _, mask_clean = cv2.threshold(mask, 60, 255, cv2.THRESH_BINARY)
    mask_clean = cv2.GaussianBlur(mask_clean, (31, 31), 0)

    # Apply white background
    mask_norm  = mask_clean / 255.0
    white_bg   = np.ones_like(sketch) * 255
    result     = (sketch * mask_norm + white_bg * (1 - mask_norm)).astype(np.uint8)

    # Final boost
    result_pil = Image.fromarray(result)
    result_pil = ImageEnhance.Contrast(result_pil).enhance(2.5)
    result_pil = ImageEnhance.Sharpness(result_pil).enhance(3.0)
    result_pil = result_pil.filter(ImageFilter.SHARPEN)

    return result_pil

# ========== UI ==========

st.title("🎨 AI Portrait Generator")
st.markdown("Upload a photo and choose your portrait style.")

# ---- Style selector ----
style = st.radio(
    "Choose a style:",
    ["✏️ Pencil Sketch", "🖼️ AI Draw"],
    horizontal=True
)

# ---- Upload ----
uploaded_file = st.file_uploader(
    "Upload your photo",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert('RGB')
    image_resized = image.resize((512, 512), Image.LANCZOS)
    orig_np = np.array(image_resized)

    st.image(image, caption="Your uploaded photo", use_column_width=True)

    if st.button("🎨 Generate Portrait"):

        with st.spinner("Generating your portrait... please wait"):

            # Load model
            net = load_model()

            # Preprocess
            tensor = preprocess(image)

            # Inference
            with torch.no_grad():
                d = net(tensor)

            # Apply selected style
            if style == "✏️ Pencil Sketch":
                result = pencil_sketch(orig_np, d)
            else:
                result = ai_draw(orig_np, d)

        # Display result
        st.success("Done!")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original", use_column_width=True)
        with col2:
            st.image(result, caption=style, use_column_width=True)

        # Download button
        result_pil = result if isinstance(result, Image.Image) else Image.fromarray(result)
        from io import BytesIO
        buf = BytesIO()
        result_pil.save(buf, format="PNG")
        st.download_button(
            label="⬇️ Download Portrait",
            data=buf.getvalue(),
            file_name="portrait.png",
            mime="image/png"
        )