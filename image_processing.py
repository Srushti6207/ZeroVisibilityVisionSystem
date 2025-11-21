import cv2
import numpy as np

# ---------- Helper: Gamma Correction ----------
def apply_gamma_correction(image, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# ---------- Helper: White Balance (more natural colors) ----------
def simple_white_balance(image):
    # Convert to LAB and equalize L-channel to fix dull colors
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# ---------- Helper: Sharpen ----------
def sharpen_image(image, strength=1.0):
    blur = cv2.GaussianBlur(image, (0, 0), 3)
    sharp = cv2.addWeighted(image, 1.0 + strength, blur, -strength, 0)
    return sharp

# ---------- 1. Low-Light / Night Enhancement ----------
def enhance_low_light(image):
    # Resize very large images to speed up
    h, w = image.shape[:2]
    if max(h, w) > 1500:
        scale = 1500 / max(h, w)
        image = cv2.resize(image, (int(w*scale), int(h*scale)))

    # 1) White balance
    balanced = simple_white_balance(image)

    # 2) LAB + CLAHE on L channel
    lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # 3) Adaptive gamma based on brightness
    mean_brightness = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY).mean()
    if mean_brightness < 60:
        gamma = 1.8
    elif mean_brightness < 90:
        gamma = 1.5
    else:
        gamma = 1.2

    enhanced = apply_gamma_correction(enhanced, gamma=gamma)

    # 4) Slight sharpen to add clarity
    enhanced = sharpen_image(enhanced, strength=0.4)

    return enhanced

# ---------- 2. Fog / Haze Removal (Improved Dehazing) ----------
def remove_fog(image):
    # Convert to float [0,1]
    img = image.astype('float32') / 255.0

    # Dark channel
    kernel_size = 15
    min_channel = cv2.min(cv2.min(img[:, :, 0], img[:, :, 1]), img[:, :, 2])
    dark = cv2.erode(min_channel, np.ones((kernel_size, kernel_size)))

    # Atmospheric light A
    flat_dark = dark.ravel()
    num_pixels = flat_dark.size
    num_bright = int(max(num_pixels * 0.001, 1))
    indices = np.argpartition(flat_dark, -num_bright)[-num_bright:]
    brightest = img.reshape(-1, 3)[indices]
    A = brightest.mean(axis=0)

    # Transmission estimate
    omega = 0.95
    transmission = 1 - omega * dark / (A.max() + 1e-6)

    # Refine transmission with guided filter (edge preservation)
    gray = cv2.cvtColor((img * 255).astype('uint8'), cv2.COLOR_BGR2GRAY)
    transmission = cv2.ximgproc.guidedFilter(
        guide=gray, src=(transmission * 255).astype('uint8'),
        radius=40, eps=1e-3
    ).astype('float32') / 255.0

    transmission = np.clip(transmission, 0.2, 1.0)

    # Recover scene radiance
    t = transmission[:, :, np.newaxis]
    J = (img - A) / t + A
    J = np.clip(J, 0, 1)

    dehazed = (J * 255).astype('uint8')

    # White balance + sharpen for more realistic look
    dehazed = simple_white_balance(dehazed)
    dehazed = sharpen_image(dehazed, strength=0.5)

    return dehazed

# ---------- 3. Smoke / Noise Removal (Cleaner + Sharper) ----------
def remove_smoke_noise(image):
    # 1) White balance to fix color cast
    balanced = simple_white_balance(image)

    # 2) Bilateral filter – preserves edges
    smooth = cv2.bilateralFilter(balanced, d=9, sigmaColor=75, sigmaSpace=75)

    # 3) Non-local means denoising
    denoised = cv2.fastNlMeansDenoisingColored(
        smooth, None,
        h=8, hColor=8,
        templateWindowSize=7,
        searchWindowSize=21
    )

    # 4) Slight sharpen so it doesn't look too soft
    final = sharpen_image(denoised, strength=0.3)

    return final
