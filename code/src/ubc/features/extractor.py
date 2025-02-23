import torch
from PIL import Image
import os
import numpy as np
import glob
from transformers import AutoImageProcessor, ViTModel
from torchvision import transforms
from tqdm.auto import tqdm
from torchvision.models.resnet import Bottleneck, ResNet
from timm.models.vision_transformer import VisionTransformer
import torch.nn as nn
from ubc.features.resnet import resnet50


def generate_tiles(img, tile_size, bgcolor=(0, 0, 0)):
    tiles = []
    # Tile image
    h, w = tile_size
    image_height, image_width = img.shape[0], img.shape[1]
    idxs = [(y, y + h, x, x + w) for y in range(0, image_height, h) for x in range(0, image_width, w)]
    image = Image.fromarray(img)
    for uid, (y, y_, x, x_) in enumerate(idxs):
        x1, y1, x2, y2 = x, y, x + min(w, image_width - x), y + min(h, image_height - y)
        tile = image.crop((x1, y1, x2, y2))
        # Make sure the tile is hxw
        if tile.size != (w, h):
            tile_ = Image.new(tile.mode, (w, h), bgcolor)
            tile_.paste(tile, (0, 0))
            tile = tile_
        tiles.append(np.array(tile))
    return tiles


def create_ctranspath_model(model_weights, imgsz=224):
    from timm054.models.layers.helpers import to_2tuple
    import torch.nn as nn

    class ConvStem(nn.Module):

        def __init__(self, img_size=imgsz, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, flatten=True,
                     output_fmt=None):
            super().__init__()

            assert patch_size == 4
            assert embed_dim % 8 == 0

            img_size = to_2tuple(img_size)

            patch_size = to_2tuple(patch_size)
            self.img_size = img_size
            self.patch_size = patch_size
            self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
            self.flatten = flatten

            stem = []
            input_dim, output_dim = 3, embed_dim // 8
            for l in range(2):
                stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
                stem.append(nn.BatchNorm2d(output_dim))
                stem.append(nn.ReLU(inplace=True))
                input_dim = output_dim
                output_dim *= 2
            stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
            self.proj = nn.Sequential(*stem)

            self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        def forward(self, x):
            B, C, H, W = x.shape
            assert H == self.img_size[0] and W == self.img_size[1], \
                f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
            x = self.proj(x)
            if self.flatten:
                x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
            x = self.norm(x)
            return x

    def ctranspath():
        model = timm054.create_model('swin_tiny_patch4_window7_224', embed_layer=ConvStem, pretrained=False)
        return model

    # Feature extractor
    model_vit = ctranspath()
    model_vit.head = nn.Identity()

    if model_weights is not None:
        print("Loading:", model_weights) if os.path.exists(model_weights) else print("Cannot load:", model_weights)
        td = torch.load(model_weights)
        model_vit.load_state_dict(td['model'], strict=True)
    model_vit.eval()

    return model_vit


def features_from_tile_ctranspath(model, img, preprocess_image=None, prepare_feed=None, is_tma=False, device="cuda"):
    # Numpy array expected (img=tile)
    img = prepare_feed(img)
    img = torch.unsqueeze(img, 0).to(device)
    with torch.no_grad():
        feat = model(img)
        feat = feat.squeeze()
    return feat


def load_ctranspath_model(weights, device="cuda"):
    prepare_feed_ctranspath = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    model = create_ctranspath_model(weights).to(device)
    return model, prepare_feed_ctranspath


def ctranspath_features(folder, target_folder, weights="./models/ctranspath/ctranspath-swin224.pth", device="cuda", tile_size=(224, 224)):

    files = glob.glob(folder + "/*/*.png")

    model, prepare_feed_ctranspath = load_ctranspath_model(weights, device=device)

    for file in tqdm(files, total=len(files)):
        # Read image
        image = np.array(Image.open(file))
        # Tile image
        tiles = generate_tiles(image, tile_size=tile_size)

        image_id = int(file.split("/")[-2])
        if target_folder is not None:
            feat_dir = os.path.join(target_folder, "%s" % image_id)
            feat_file = os.path.join(feat_dir, os.path.basename(file).replace(".png", "_%d.npz" % len(tiles)))
            if os.path.exists(feat_file):
                continue

        # Extract feature for each tile after normalization
        tiles_features = []
        for tile in tiles:
            tile_feature = features_from_tile_ctranspath(model, tile, prepare_feed=prepare_feed_ctranspath, device=device)
            tiles_features.append(tile_feature)
        tiles_features = torch.stack(tiles_features, dim=0)  # (N_TILES, 768)
        tiles_features = tiles_features.cpu().numpy()

        assert(len(tiles_features) == len(tiles)), "total not match %d %d for %s" % (len(tiles_features), len(tiles), file)
        if target_folder is not None:
            feat_dir = os.path.join(target_folder, "%s"%image_id)
            os.makedirs(feat_dir, exist_ok=True)
            feat_file = os.path.join(feat_dir, os.path.basename(file).replace(".png", "_%d.npz" % len(tiles)))
            with open(feat_file, "wb") as outfile:
                np.savez(outfile, features=tiles_features)


class ResNetTrunk(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.fc  # remove FC layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class RetCCL(nn.Module):
    def __init__(self, weights):
        super().__init__()

        self.model = resnet50(num_classes=128, mlp=False, two_branch=False, normlinear=True)
        self.model.fc = nn.Identity()

        self.model.load_state_dict(torch.load(weights, map_location=torch.device("cpu")), strict=True)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def get_pretrained_url(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "BT": "bt_rn50_ep200.torch",
        "MoCoV2": "mocov2_rn50_ep200.torch",
        "SwAV": "swav_rn50_ep200.torch",
        "DINO_p16": "dino_vit_small_patch16_ep200.torch",
        "DINO_p8": "dino_vit_small_patch8_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url


def resnet50t(pretrained, progress, key, **kwargs):
    model = ResNetTrunk(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_url = get_pretrained_url(key)
        verbose = model.load_state_dict(torch.hub.load_state_dict_from_url(pretrained_url, progress=progress))
        print(verbose)
    return model


def vit_small(pretrained, progress, key, **kwargs):
    patch_size = kwargs.get("patch_size", 16)
    model = VisionTransformer(
        img_size=224, patch_size=patch_size, embed_dim=384, num_heads=6, num_classes=0
    )
    if pretrained:
        pretrained_url = get_pretrained_url(key)
        verbose = model.load_state_dict(
            torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
        )
        print(verbose)
    return model


def features_from_tile_lunitdino(model, img, preprocess_image=None, prepare_feed=None, is_tma=False, device="cuda"):
    # Numpy array expected (img=tile)
    img = prepare_feed(img)
    img = torch.unsqueeze(img, 0).to(device)
    # print("img", img.shape, img.min(), img.max())
    with torch.no_grad():
        feat = model(img)
        feat = feat.squeeze()
        # print("feat", feat.shape, feat.min(), feat.max())
    return feat


def load_lunitdino_model(weights, device="cuda"):
    # https://github.com/lunit-io/benchmark-ssl-pathology/releases/tag/pretrained-weights
    LUNIT_MEAN = [0.70322989, 0.53606487, 0.66096631]
    LUNIT_STD = [0.21716536, 0.26081574, 0.20723464]

    prepare_feed_lunitdino = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=LUNIT_MEAN, std=LUNIT_STD)
    ])

    # initialize ViT-S/16 trunk using DINO pre-trained weight
    if weights == "DINO_p16":
        model = vit_small(pretrained=True, progress=False, key=weights, patch_size=16)
    elif weights == "BT":
        model = resnet50t(pretrained=True, progress=False, key="BT")
    elif weights == "SwAV":
        model = resnet50t(pretrained=True, progress=False, key="SwAV")

    model = model.eval()
    model = model.to(device)

    return model, prepare_feed_lunitdino


def load_retccl_model(weights, device="cuda"):

    prepare_feed_retccl = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if os.path.exists(weights):
        model = RetCCL(weights)

        model = model.eval()
        model = model.to(device)
    else:
        raise Exception("Weights not found:", weights)

    return model, prepare_feed_retccl


def lunitdino_features(folder, target_folder, weights="DINO_p16", device="cuda", tile_size=(224, 224)):
    files = glob.glob(folder + "/*/*.png")

    model, prepare_feed_lunitdino = load_lunitdino_model(weights, device=device)

    for file in tqdm(files, total=len(files)):
        # Read image
        image = np.array(Image.open(file))
        # Tile image
        tiles = generate_tiles(image, tile_size=tile_size)

        image_id = int(file.split("/")[-2])
        if target_folder is not None:
            feat_dir = os.path.join(target_folder, "%s" % image_id)
            feat_file = os.path.join(feat_dir, os.path.basename(file).replace(".png", "_%d.npz" % len(tiles)))
            if os.path.exists(feat_file):
                continue

        # Extract feature for each tile after normalization
        tiles_features = []
        for tile in tiles:
            tile_feature = features_from_tile_lunitdino(model, tile, prepare_feed=prepare_feed_lunitdino, device=device)
            tiles_features.append(tile_feature)
        tiles_features = torch.stack(tiles_features, dim=0)  # (N_TILES, 768)
        tiles_features = tiles_features.cpu().numpy()

        assert (len(tiles_features) == len(tiles)), "total not match %d %d for %s" % (
            len(tiles_features), len(tiles), file)
        if target_folder is not None:
            feat_dir = os.path.join(target_folder, "%s" % image_id)
            os.makedirs(feat_dir, exist_ok=True)
            feat_file = os.path.join(feat_dir, os.path.basename(file).replace(".png", "_%d.npz" % len(tiles)))
            with open(feat_file, "wb") as outfile:
                np.savez(outfile, features=tiles_features)

