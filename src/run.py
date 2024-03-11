import torch
from my_utils import CustomCocoDataset, get_transform, get_model_instance_segmentation
from references.engine import train_one_epoch, evaluate
from references import utils

from torchvision.io import read_image
from torchvision.io import ImageReadMode

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 91 + 1  # including background

    image = read_image('../data/images/000000000139.jpg', ImageReadMode.RGB)
    images = [image]

    model = get_model_instance_segmentation(num_classes)
    model.to(device)
    model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    output = model(x)
    print(type(output))
    print()
