# Zero-Visibility Vision System

A Flask web application for improving photos captured in poor-visibility conditions. Upload an image, choose the type of degradation, and download an enhanced result.

> This is a classical computer-vision demo using OpenCV image-processing techniques; it does not use a trained AI model.

## Features

- **Low-light / night enhancement** — white balancing, local contrast enhancement (CLAHE), adaptive gamma correction, and sharpening.
- **Fog / haze removal** — dark-channel dehazing, guided-filter transmission refinement, white balancing, and sharpening.
- **Smoke / noise reduction** — bilateral filtering, non-local-means denoising, and edge-preserving sharpening.
- Side-by-side original and processed image preview.
- Downloadable PNG output.

## Tech stack

- Python 3
- Flask
- OpenCV with contrib modules
- NumPy
- HTML/CSS

## Project structure

```text
ZeroVisibilityVisionSystem/
├── app.py                 # Flask routes and upload/result handling
├── image_processing.py    # Enhancement pipelines
├── index.html             # Web-page template (move to templates/ before running)
├── fog img.jpg            # Sample image
└── README.md
```

At runtime, the application creates these directories when needed:

```text
static/
├── uploads/               # Uploaded source images
└── results/               # Generated enhanced images
```

## Getting started

### 1. Clone the repository

```bash
git clone <repository-url>
cd ZeroVisibilityVisionSystem
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
```

**Windows (PowerShell)**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install Flask numpy opencv-contrib-python
```

`opencv-contrib-python` is required because the fog-removal pipeline uses OpenCV's `ximgproc.guidedFilter` module.

### 4. Put the Flask template in the expected directory

Flask loads templates from a `templates` directory. Create it and move `index.html` into it:

**Windows (PowerShell)**

```powershell
New-Item -ItemType Directory -Force templates
Move-Item index.html templates\index.html
```

**macOS/Linux**

```bash
mkdir -p templates
mv index.html templates/index.html
```

### 5. Run the application

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## How to use it

1. Choose a JPG or PNG image.
2. Select one enhancement mode:
   - **Low-Light / Night** for dark images.
   - **Fog / Haze** for hazy or foggy scenes.
   - **Smoke / Noise** for noisy or smoke-affected images.
3. Select **Enhance Vision**.
4. Compare the original and enhanced images, then download the result if desired.

## Notes and limitations

- Images whose longest side exceeds 1,500 pixels are resized in low-light mode to improve processing speed.
- Enhancement quality depends on the source image and the selected mode; it cannot reconstruct scene detail that is not present in the original capture.
- Uploaded and generated files are stored locally in `static/uploads/` and `static/results/`. Clean these folders periodically if the app is used frequently.
- The development server runs with Flask debug mode enabled. Disable debug mode before deploying the app publicly.

## License

No license is currently specified for this project.
