import numpy as np
import albumentations as A
import random
import cv2


# https://pyimagesearch.com/2017/06/05/computing-image-colorfulness-with-opencv-and-python/
def image_colorfulness(image: np.ndarray) -> float:
    """
    Args:
        image: a single image in RGB format

    Returns:
        float: colorfulness
    """
    r, g, b = np.rollaxis(image.astype(float), 2)
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    rb_mean = np.mean(rg)
    rb_std = np.std(rg)
    yb_mean = np.mean(yb)
    yb_std = np.std(yb)
    std_root = np.sqrt((rb_std ** 2) + (yb_std ** 2))
    mean_root = np.sqrt((rb_mean ** 2) + (yb_mean ** 2))
    return std_root + (0.3 * mean_root)


class SimulateTMA(A.DualTransform):

    def __init__(self, std, radius_ratio=(0.9, 1.0), ellipse_ratio=(0.9, 1.0), angle=(-90., 90.), background_color=(-1, -1, -1), background_color_ratio=1.0, noise_level=(0.0, 0.0), black_replacement_color=None, always_apply=False, p=1.0):
        super(SimulateTMA, self).__init__(always_apply, p)
        self.std = std
        self.radius_ratio = radius_ratio
        self.ellipse_ratio = ellipse_ratio
        self.background_color = background_color
        self.background_color_ratio = background_color_ratio
        self.angle = angle
        self.noise_level = noise_level
        self.black_replacement_color = black_replacement_color

    def apply(self, img, **params):
        height, width = img.shape[:2]
        # Replace the black regions with the replacement color
        if self.black_replacement_color is not None:
            black_mask = np.all(img == [0, 0, 0], axis=-1)
            img[black_mask] = self.black_replacement_color
        # img_std = image_colorfulness(img) # (15, 40)
        img_std = np.std(img) if self.std[0] != -1 else 0  # (20, 50)
        if (self.std[0] == -1) or ((img_std <= self.std[1]) and (img_std >= self.std[0])):
            # Draw circle
            x_center = width // 2
            y_center = height // 2
            radius_w = int((width//2)*random.uniform(self.radius_ratio[0], self.radius_ratio[1]))  # Random radius
            radius_h = int(radius_w*random.uniform(self.ellipse_ratio[0], self.ellipse_ratio[1]))  # int((height//2)*random.uniform(self.radius_ratio[0], self.radius_ratio[1]))  # Random radius
            angle = int(random.uniform(self.angle[0], self.angle[1]))
            mask = cv2.ellipse(np.zeros_like(img), (x_center, y_center), (radius_w, radius_h), angle, 0, 360, color=(255, 255, 255), thickness=-1)
            # Add noise to the contour to mimic TMA
            if self.noise_level[1] > 0:
                contour = cv2.findContours(mask[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                contour_with_noise = contour[0] + np.random.randint(-self.noise_level[0], self.noise_level[1], contour[0].shape)
                int(random.uniform(self.angle[0], self.angle[1]))
                mask = cv2.drawContours(np.zeros_like(img), [contour_with_noise], -1, (255, 255, 255), -1)
            # Apply masks
            inverse_mask = cv2.bitwise_not(mask)
            # Background color
            bg_color = self.background_color
            if self.background_color == (-1, -1, -1):
                bg_ratio = random.uniform(self.background_color_ratio[0], self.background_color_ratio[1])
                bg_color = tuple((np.max(img, axis=(0,1))*bg_ratio).astype(np.uint8)) # Auto color
            color_outside_circle = np.zeros_like(img) # Black image
            color_outside_circle[:] = bg_color
            color_outside_circle = cv2.bitwise_and(color_outside_circle, inverse_mask)
            img = cv2.bitwise_and(img, mask)
            img = cv2.add(img, color_outside_circle)
        return img

    def get_transform_init_args_names(self):
        return ("std", "radius_ratio", "ellipse_ratio", "angle", "background_color", "background_color_ratio", "noise_level", "black_replacement_color")
