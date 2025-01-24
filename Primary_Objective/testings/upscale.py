import cv2
from PIL import Image

def upscale_image(image_path, scale=2):
    img = cv2.imread(image_path)
    img_upscaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(cv2.cvtColor(img_upscaled, cv2.COLOR_BGR2RGB))

# Example: Upscale and visualize
upscaled_image = upscale_image(r"C:\Users\Mony\Downloads\tango-cv-assessment-dataset\0225_c1s1_054851_00.jpg")
upscaled_image.show()
