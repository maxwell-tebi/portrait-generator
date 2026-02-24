import os
import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
from skimage import transform as skimage_transform
from huggingface_hub import hf_hub_download
from u2net_model import U2NET

# ---- Page config ----
st.set_page_config(
    page_title="AI Portrait Generator",
    page_icon="🎨",
    layout="centered"
)

# ---- Hugging Face model config ----
HF_REPO_ID  = "Maxwelltebi/u2net-portrait"
HF_FILENAME = "u2net_portrait.pth"

# ---- Load model ----
@st.cache_resource
def load_model():
    with st.spinner("Loading model... (first time only, ~170MB)"):
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            repo_type="model"
        )
    net = U2NET(3, 1)
    net.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
    net.eval()
    return net

# ---- Helper: normalize prediction ----
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    return (d - mi) / (ma - mi)

# ---- Helper: preprocess image ----
def preprocess(image):
    img = np.array(image.convert('RGB'))
    img = skimage_transform.resize(img, (512, 512), mode='constant')
    img = img / np.max(img)
    tmp = np.zeros((512, 512, 3))
    tmp[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
    tmp[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
    tmp[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225
    tmp = tmp.transpose((2, 0, 1))
    return torch.from_numpy(tmp).unsqueeze(0).float()

# ---- Style A: AI Draw ----
def ai_draw(orig_np, d):
    pred     = 1.0 - normPRED(d[:, 0, :, :])
    pred     = normPRED(pred)
    mask     = pred.squeeze().cpu().data.numpy()
    portrait = Image.fromarray((mask * 255)).convert('RGB')
    return portrait

# ---- Style B: Detailed Pencil Sketch with Shading ----
def pencil_sketch(orig_np, d):

    # ---- Step 1: Smooth the image first to kill noise ----
    # Bilateral filter preserves edges while removing noise/dots
    smoothed = orig_np.copy()
    for _ in range(4):
        smoothed = cv2.bilateralFilter(smoothed, d=9, sigmaColor=50, sigmaSpace=50)

    # ---- Step 2: Convert to grayscale ----
    gray = cv2.cvtColor(smoothed, cv2.COLOR_RGB2GRAY)

    # ---- Step 3: Build shading layer (soft tonal shading) ----
    # Invert and blur heavily for smooth shading gradients
    gray_inv       = cv2.bitwise_not(gray)
    blur_light     = cv2.GaussianBlur(gray_inv, (21, 21), 0)
    blur_heavy     = cv2.GaussianBlur(gray_inv, (61, 61), 0)

    # Dodge blend with light blur = fine detail shading
    shade_fine     = cv2.divide(gray, cv2.bitwise_not(blur_light), scale=256.0)

    # Dodge blend with heavy blur = broad tonal shading
    shade_broad    = cv2.divide(gray, cv2.bitwise_not(blur_heavy), scale=256.0)

    # Blend both shading layers together
    shading        = cv2.addWeighted(shade_fine, 0.5, shade_broad, 0.5, 0)

    # ---- Step 4: Build clean edge lines (no dots/noise) ----
    # Use Canny on the smoothed image — clean lines only
    edges          = cv2.Canny(gray, 20, 80)

    # Dilate edges slightly so lines are bold and visible
    kernel         = np.ones((2, 2), np.uint8)
    edges          = cv2.dilate(edges, kernel, iterations=1)

    # Blur edges slightly to make them look hand-drawn not digital
    edges          = cv2.GaussianBlur(edges, (3, 3), 0)

    # Convert edges to white-background format (dark lines on white)
    edges_inv      = cv2.bitwise_not(edges)

    # ---- Step 5: Combine shading + edges ----
    # Multiply blends: dark shading stays dark, white areas stay light
    sketch         = cv2.multiply(
        shading.astype(np.float32),
        edges_inv.astype(np.float32),
        scale=1/255.0
    ).astype(np.uint8)

    # ---- Step 6: Saliency mask — isolate subject, clean background ----
    pred           = normPRED(d[:, 0, :, :])
    mask_np        = pred.squeeze().cpu().data.numpy()
    mask           = cv2.resize((mask_np * 255).astype(np.uint8), (512, 512))

    # Smooth mask edges for natural subject cutout
    mask           = cv2.GaussianBlur(mask, (21, 21), 0)
    _, mask_clean  = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
    mask_clean     = cv2.GaussianBlur(mask_clean, (41, 41), 0)
    mask_norm      = mask_clean / 255.0

    # Apply pure white background outside subject
    white_bg       = np.ones_like(sketch) * 255
    result         = (sketch * mask_norm + white_bg * (1 - mask_norm)).astype(np.uint8)

    # ---- Step 7: Final enhancement ----
    result_pil     = Image.fromarray(result)

    # Boost contrast to make shading pop
    result_pil     = ImageEnhance.Contrast(result_pil).enhance(1.8)

    # Sharpen to make lines crisp
    result_pil     = ImageEnhance.Sharpness(result_pil).enhance(2.5)

    # One smooth pass to remove any remaining grain
    result_pil     = result_pil.filter(ImageFilter.SMOOTH)

    return result_pil


# ========== UI ==========

st.title("🎨 AI Portrait Generator")
st.markdown("Upload a photo and choose your portrait style.")
st.markdown("---")

style = st.radio(
    "Choose a style:",
    ["✏️ Pencil Sketch", "🖼️ AI Draw"],
    horizontal=True
)

uploaded_file = st.file_uploader(
    "Upload your photo",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image         = Image.open(uploaded_file).convert('RGB')
    image_resized = image.resize((512, 512), Image.LANCZOS)
    orig_np       = np.array(image_resized)

    st.image(image, caption="Your uploaded photo", use_column_width=True)
    st.markdown("---")

    if st.button("🎨 Generate Portrait"):

        with st.spinner("Generating your portrait... please wait"):
            net    = load_model()
            tensor = preprocess(image)
            with torch.no_grad():
                d = net(tensor)
            if style == "✏️ Pencil Sketch":
                result = pencil_sketch(orig_np, d)
            else:
                result = ai_draw(orig_np, d)

        st.success("✅ Portrait generated!")
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original", use_column_width=True)
        with col2:
            st.image(result, caption=style, use_column_width=True)

        from io import BytesIO
        result_pil = result if isinstance(result, Image.Image) else Image.fromarray(result)
        buf = BytesIO()
        result_pil.save(buf, format="PNG")
        st.download_button(
            label="⬇️ Download Portrait",
            data=buf.getvalue(),
            file_name="portrait.png",
            mime="image/png"
        )