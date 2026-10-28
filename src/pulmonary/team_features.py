"""Shared training/serving transforms and text features for the team model."""

from PIL import Image, ImageOps
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from torch import nn
from torchvision import models, transforms

from pulmonary.team_rules import make_report_text, t29_clean_text, text_meta_features

PREPROCESS = "team29-gray-lanczos-letterbox384-imagenet-v1"
SIZE = 384


def letterbox(image):
    image = ImageOps.exif_transpose(image).convert("L")
    ratio = SIZE / max(image.size)
    image = image.resize(
        tuple(max(1, round(v * ratio)) for v in image.size), Image.Resampling.LANCZOS
    )
    canvas = Image.new("L", (SIZE, SIZE), 0)
    canvas.paste(image, ((SIZE - image.width) // 2, (SIZE - image.height) // 2))
    return canvas


def text_frame(frame):
    result = frame.copy()
    result["text"] = result.apply(make_report_text, axis=1)
    result["indication_clean_t29"] = result.indication.map(t29_clean_text)
    result["comparison_clean_t29"] = result.comparison.map(t29_clean_text)
    return result


class TeamTextModel:
    """Persist every fitted component missing from the original notebook export."""

    def __init__(self):
        self.word = TfidfVectorizer(
            ngram_range=(1, 2), min_df=3, max_df=0.9, max_features=5000, sublinear_tf=True
        )
        self.char = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=5000, sublinear_tf=True
        )
        self.scaler = StandardScaler()

    def matrix(self, frame, fit=False):
        frame = text_frame(frame)
        if fit:
            self.word.fit(frame.text)
            self.char.fit(frame.text)
            self.scaler.fit(text_meta_features(frame))
        return hstack(
            [
                self.word.transform(frame.text),
                self.char.transform(frame.text),
                csr_matrix(self.scaler.transform(text_meta_features(frame))),
            ]
        ).tocsr()

    def predict_proba(self, frame):
        return self.classifier.predict_proba(self.matrix(frame))


def image_transforms(augment=False):
    steps = (
        [
            transforms.RandomRotation(5),
            transforms.RandomAffine(0, translate=(0.02, 0.02), scale=(0.97, 1.03)),
        ]
        if augment
        else []
    )
    return transforms.Compose(
        steps
        + [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def build_model(name, pretrained=False):
    if name == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        net.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(net.fc.in_features, 2))
    elif name == "densenet121":
        net = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
        net.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(net.classifier.in_features, 2))
    else:
        raise ValueError("Selected team fusion requires resnet50 and densenet121")
    return net
