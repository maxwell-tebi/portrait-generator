# AI Portrait Generator — U2-Net

A deep learning web application that transforms human photos into artistic portraits using the U-squared Net (U2-Net) architecture. Users can upload a photo and choose between two styles: AI Draw and Pencil Sketch.

**Live App**: https://portrait-generator.streamlit.app/ 

---



## Overview

This project implements the U2-Net architecture from scratch in PyTorch, loads pretrained weights, and serves the model through a Streamlit web interface. The model performs Salient Object Detection (SOD) — isolating the most visually significant subject in an image — and the portrait styles are applied as post-processing on top of the resulting saliency map.

The pretrained weights are hosted on Hugging Face Hub and downloaded automatically on first run.

---

## Architecture

U2-Net is a two-level nested U-structure designed specifically for Salient Object Detection. Unlike segmentation models that borrow classification backbones (VGG, ResNet), U2-Net was purpose-built to capture both local detail and global context simultaneously.

### Why U2-Net?

Most SOD networks at the time relied on feature extractors originally trained for image classification. These architectures optimise for "what is in this image" rather than "where exactly is the subject and what are its precise boundaries." U2-Net addresses this by introducing the Residual U-block (RSU).

### Residual U-block (RSU)

Each RSU block is a complete encoder-decoder with skip connections nested inside a single stage of the outer network. This means multi-scale feature extraction happens at every stage internally, before the outer U-Net aggregates features across stages.

```
Input
  └── Input convolution (local feature extraction)
        └── Encoder (progressive downsampling)
              └── Decoder (progressive upsampling + skip connections)
  └── Residual connection (input + decoded output)
```

The depth of each RSU block decreases as we go deeper into the network:

| Block  | Depth | Location         | Notes                                  |
|--------|-------|------------------|----------------------------------------|
| RSU-7  | 7     | Encoder stage 1  | Highest resolution, maximum context   |
| RSU-6  | 6     | Encoder stage 2  |                                        |
| RSU-5  | 5     | Encoder stage 3  |                                        |
| RSU-4  | 4     | Encoder stage 4  |                                        |
| RSU-4F | 4     | Stages 5, 6      | Fully dilated — no pooling operations |

RSU-4F replaces all pooling and upsampling with dilated convolutions at the deepest levels, preserving spatial resolution while expanding the receptive field.

### Full Network

The complete U2-Net consists of:

- 6-stage encoder (RSU-7 → RSU-6 → RSU-5 → RSU-4 → RSU-4F → RSU-4F)
- 5-stage decoder (RSU-4F → RSU-4 → RSU-5 → RSU-6 → RSU-7)
- 6 side output convolutions producing intermediate saliency maps
- 1 final fusion convolution combining all side outputs
- Sigmoid activation on the final output

Total parameters: **44,973,473**

---

## Portrait Styles

### AI Draw
Uses the raw saliency probability map output from U2-Net. The model's detection of the salient subject is inverted and normalised to produce a high-contrast black and white drawing effect.

### Pencil Sketch
A multi-step post-processing pipeline applied on top of the saliency mask:

1. The saliency mask isolates the subject and removes the background
2. A dodge blend between the grayscale image and its blurred inverse produces tonal shading
3. Canny edge detection on a pre-blurred grayscale extracts clean structural lines
4. Shading and edge layers are combined via multiply blend
5. Final contrast and sharpness enhancement

---

## Project Structure

```
portrait-generator/
│
├── app.py                  # Streamlit web application
├── u2net_model.py          # Full U2-Net architecture (REBNCONV, RSU blocks, U2NET)
├── requirements.txt        # Python dependencies
├── .gitignore              # Excludes model weights and virtual environment
└── README.md               # This file

# Not tracked by git:
├── u2net_portrait.pth      # Pretrained weights (hosted on Hugging Face Hub)
└── venv/                   # Virtual environment
```

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/portrait-generator.git
cd portrait-generator

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run Locally

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

On first run, the pretrained weights (~170MB) will be downloaded automatically from Hugging Face Hub and cached locally. Subsequent runs load from cache instantly.

### Usage

1. Select a portrait style — Pencil Sketch or AI Draw
2. Upload a photo (JPG or PNG)
3. Click Generate Portrait
4. Download the result using the download button

---

## Deployment

The app is deployed on Streamlit Cloud. The model weights are hosted on Hugging Face Hub at `Maxwelltebi/u2net-portrait` and downloaded at runtime using the `huggingface_hub` library.

To deploy your own instance:

1. Fork this repository
2. Upload your own `u2net_portrait.pth` to a Hugging Face model repository
3. Update `HF_REPO_ID` in `app.py` to point to your repository
4. Connect the forked repository to Streamlit Cloud at share.streamlit.io

---

## Dependencies

| Package               | Purpose                            |
|-----------------------|------------------------------------|
| torch / torchvision   | Model architecture and inference   |
| streamlit             | Web application interface          |
| opencv-python         | Image processing and sketch effect |
| Pillow                | Image enhancement                  |
| scikit-image          | Image resizing and transformation  |
| huggingface_hub       | Model weight download and caching  |
| numpy                 | Numerical operations               |



## License

MIT License
