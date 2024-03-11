import torch
from my_utils import CustomCocoDataset, get_transform, get_model_instance_segmentation
from references.engine import train_one_epoch, evaluate
from references import utils


if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 91 + 1  # including background
    dataset = CustomCocoDataset('../data/annotations/instances_val2017.json', '../data/images', get_transform(train=True))
    # dataset_test = CustomCocoDataset('../data/annotations/instances_val2017.json', '../data/images', get_transform(train=False))

    # split the dataset in train and test set
    # indices = torch.randperm(len(dataset)).tolist()
    # dataset = torch.utils.data.Subset(dataset, indices[:-5])
    # dataset_test = torch.utils.data.Subset(dataset_test, indices[-5:])

    n_samples = len(dataset)
    print(n_samples)
    train_size = int(n_samples * 0.8) 
    val_size = n_samples - train_size 

    # shuffleしてから分割
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    val_dataset.transform = None  # validation では transform は不要

    # define training and validation data loaders
    train_data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        # num_workers=4,  # TODO 要調査: これがあるとエラーが出る
        collate_fn=utils.collate_fn  # TODO 要調査: これがないとエラーが出る
    )

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        # num_workers=4,  # TODO 要調査: これがあるとエラーが出る
        collate_fn=utils.collate_fn  # TODO 要調査: これがないとエラーが出る
    )

    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )

    print("Start training.")
    num_epochs = 2
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, val_data_loader, device=device)
    print("End of training.")