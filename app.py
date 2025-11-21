import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for
import cv2

from image_processing import enhance_low_light, remove_fog, remove_smoke_noise

app = Flask(__name__)

# Configure upload folders
UPLOAD_FOLDER = os.path.join('static', 'uploads')
RESULT_FOLDER = os.path.join('static', 'results')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER


def save_image(image, folder, prefix="img"):
    """Helper to save numpy image and return relative path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{prefix}_{timestamp}.png"
    path = os.path.join(folder, filename)
    cv2.imwrite(path, image)
    return path  # relative path


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html',
                           original_image=None,
                           processed_image=None,
                           mode=None)


@app.route('/enhance', methods=['POST'])
def enhance():
    if 'image' not in request.files:
        return redirect(url_for('index'))

    file = request.files['image']
    mode = request.form.get('mode')

    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        # Save original file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Read using OpenCV
        image = cv2.imread(filepath)
        if image is None:
            return "Error: Unable to read image. Please upload a valid image file."

        # Process based on mode
        if mode == 'low_light':
            processed = enhance_low_light(image)
        elif mode == 'fog':
            processed = remove_fog(image)
        elif mode == 'smoke':
            processed = remove_smoke_noise(image)
        else:
            processed = image  # fallback

        # Save processed image
        result_path = save_image(processed, app.config['RESULT_FOLDER'], prefix="result")

        # For HTML, we need paths relative to project root
        original_rel = filepath.replace("\\", "/")
        result_rel = result_path.replace("\\", "/")

        return render_template('index.html',
                               original_image=original_rel,
                               processed_image=result_rel,
                               mode=mode)

    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
