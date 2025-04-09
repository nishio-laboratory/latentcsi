import sys
import cv2
import numpy as np
from PIL import Image

# Load Haar Cascade for face detection.


# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

def blur_faces_opencv(image: Image.Image) -> Image.Image:
    """
    Detects faces in a PIL image using Haar cascades and applies Gaussian blur.

    Args:
        image: A 512x512 RGB PIL Image.

    Returns:
        A new PIL Image with blurred faces.
    """
    # Convert the PIL image to a numpy array (RGB).
    img_np = np.array(image)
    # Convert to grayscale required for Haar detection.
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Detect faces.
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, )

    for (x, y, w, h) in faces:
        # Extend the region by 5 pixels in every direction.
        x_min = max(x - 5, 0)
        y_min = max(y - 5, 0)
        x_max = min(x + w + 5, img_np.shape[1])
        y_max = min(y + h + 5, img_np.shape[0])
        face_region = img_np[y_min:y_max, x_min:x_max]
        face_blurred = cv2.GaussianBlur(face_region, (61, 61), 30)
        img_np[y_min:y_max, x_min:x_max] = face_blurred

    return Image.fromarray(img_np)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python blur_faces_opencv.py input_image.jpg output_image.jpg")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    image = Image.open(input_path).convert('RGB')
    blurred_image = blur_faces_opencv(image)
    blurred_image.save(output_path)
    print(f"Blurred image saved to {output_path}")
