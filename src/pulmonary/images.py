"""One deterministic image transform shared by training and serving."""

from PIL import Image, ImageOps
from torchvision import transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
PREPROCESS_VERSION = "grayscale-contain-pad-bilinear-imagenet-v1"


def prepare_pil(image, size=224):
    image = ImageOps.exif_transpose(image).convert("L")
    # ImageOps.pad crops to fit; contain + expand actually preserves the full image.
    image = ImageOps.contain(image, (size, size), method=Image.Resampling.BILINEAR)
    left = (size - image.width) // 2
    top = (size - image.height) // 2
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(image, (left, top))
    return canvas.convert("RGB")


def image_tensor(image, size=224, augment=False):
    image = prepare_pil(image, size)
    if augment:
        image = transforms.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.97, 1.03))(
            image
        )
    return transforms.Normalize(MEAN, STD)(transforms.ToTensor()(image))
