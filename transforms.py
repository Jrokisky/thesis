from torchvision import transforms

data_mean = [0.485, 0.456, 0.406]
data_std = [0.229, 0.224, 0.225]

webly_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(data_mean, data_std),
])

ss_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(data_mean, data_std),
])

resize = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def get_ss_transform(mean, std):
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std),
    ])

def get_coco_crop_transform(mean, std):
    return transforms.Compose([
        transforms.Resize((64,64)),
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(value=mean)
    ])


def get_webly_transform(mean, std):
    # Have already been rescaled to 128 x 128
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(value=mean)
    ])
