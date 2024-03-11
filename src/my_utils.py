from torchvision.io import read_image
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class CustomCocoDataset(Dataset):
    def __init__(self, annotation_path: str, image_dir: str, transform=None):
        """
        Attributes
        ----------
        annotation_path: str
            path to the annotation file
        image_dir: str
            path to the image directory
        transform:
            transform to be applied to the image
        """
        self.coco = COCO(annotation_path)
        self.image_dir = image_dir
        self.transform = transform
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index: int):
        """
        Parameters
        ----------
        index: int
            index of the image

        Returns
        -------
        img: torch.Tensor
            image tensor
        (boxes, labels, masks):
            tuple of bounding boxes and labels
        """
        # 画像IDを取得
        img_id = self.ids[index]
        # COCOライブラリを使用して画像のパスとアノテーションを取得
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        coco_anns = self.coco.loadAnns(ann_ids)
        path = self.coco.loadImgs(img_id)[0]['file_name']

        # アノテーション情報を取得
        boxes = []
        labels = []
        masks = []
        for ann in coco_anns:
            boxes.append(ann['bbox'])
            labels.append(ann['category_id'])
            masks.append(self.coco.annToMask(ann))

        # テンソルへの変換
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # 画像を読み込み
        img = read_image(os.path.join(self.image_dir, path))

        # 画像に前処理を適用（オプション）
        if self.transform is not None:
            img = self.transform(img)

        target = {"boxes": torchvision.ops.box_convert(boxes, 'xywh', 'xyxy'),
                  "labels": torch.as_tensor(labels, dtype=torch.int64),
                  "masks": tv_tensors.Mask(np.array(masks))
                  }
        # tv_tensors.Mask(np.array(masks)) で tv_tensors.Mask(masks)  とすると
        # UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. という警告が出る
        return img, target

    def __len__(self):
        return len(self.ids)


def get_transform(train: bool):
    """
    Parameters
    ----------
    train: bool
        whether the transform is for training or not

    Returns
    -------
    torchvision.transforms.v2.Compose
        composed transform
    """
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model