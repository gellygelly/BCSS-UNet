from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from unet.config import *
from unet.bcss import BCSSDataset


class InvalidModeError(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


def create_dataset_instances(mode='test'): 
    if mode not in ('train', 'val', 'test'):
        raise InvalidModeError("mode must be 'train', 'val' or 'test'")

    # Define transformations using Albumentations
    # Normalize - Prevent RuntimeError: Input type (torch.cuda.ByteTensor) and weight type (torch.cuda.FloatTensor) should be the same
    transforms_train = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        ToTensorV2(),
    ])

    transforms_val = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
    ])

    if mode == 'train':
        train_dataset = BCSSDataset(TRAIN_IMAGE_PATH, TRAIN_MASK_PATH, transform=transforms_train)
        return train_dataset

    elif mode == 'val':
        val_dataset = BCSSDataset(VAL_IMAGE_PATH, VAL_MASK_PATH, transform=transforms_val)
        return val_dataset 
    
    else: # mode = 'test
        test_dataset = BCSSDataset(TEST_IMAGE_PATH, TEST_MASK_PATH, transform=transforms_val)
        return test_dataset

def create_dataloader(dataset):
    print(f'Dataset Sample: {len(dataset)}')
    image, mask= dataset[0]
    print(image.shape, mask.shape)      

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader
