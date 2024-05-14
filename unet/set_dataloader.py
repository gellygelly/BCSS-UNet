
from torch.utils.data import DataLoader, Subset
import numpy as np
import config

# PATH
ROOT_PATH = '/data/BCSS_512_1class_UNet/'

TRAIN_IMAGE_PATH = ROOT_PATH+'train/'
VAL_IMAGE_PATH = ROOT_PATH+'val/'
TEST_IMAGE_PATH = ROOT_PATH+'test/'

TRAIN_MASK_PATH = ROOT_PATH+'train_mask/'
VAL_MASK_PATH = ROOT_PATH+'val_mask/'
TEST_MASK_PATH = ROOT_PATH+'val_mask/'

def create_dataset_instances(mode='test'): 
    if mode not in ('train', 'test'):
    raise InvalidModeError("mode must be 'train' or 'test'")

    # Define transformations using Albumentations
    transforms_train = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    transforms_val = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
    ])

    if task = 'train':
        train_dataset = BCSSDataset(TRAIN_IMAGE_PATH, TRAIN_MASK_PATH, transform=transforms_train)
        val_dataset = BCSSDataset(VAL_IMAGE_PATH, VAL_MASK_PATH, transform=transforms_val)
        return train_dataset, val_datastet

    else: # mode = 'test
        test_dataset = BCSSDataset(TEST_IMAGE_PATH, TEST_MASK_PATH, transform=transforms_val)

        return test_dataset

    return None


def create_dataloader(mode='test'):
    if mode not in ('train', 'test'):
    raise InvalidModeError("mode must be 'train' or 'test'")
    
    batch_size= 64 # TO config.py로 바꿀 것

    if mode == 'train':
        train_dataset, val_dataset = create_dataset_instances(mode='train')

        print(f'Train Sample: {len(train_dataset)}')
        print(f'Validation Sample: {len(val_dataset)}')

        # Load the first image and its mask from the training dataset and view the size
        print(image, mask = train_dataset[0])
        print(image.shape, mask.shape)      

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True) 

        return train_loader, val_loader

    else: # test
        test_dataset = create_dataset_instances(mode='test')
        test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True) 

        return test_loader

# TODO 요 부분은 나중에?
    # # Define the size of your subset
    # subset_size = 500

    # # Create random indices for the subset
    # indices = np.random.permutation(len(train_dataset))[:subset_size]

    # # Create a Subset instance
    # train_subset = Subset(train_dataset, indices)

    # # Now use this subset to create a DataLoader
    # train_loader_subset = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

    # # Similarly for validation data if needed
    # val_indices = np.random.permutation(len(val_dataset))[:subset_size]
    # val_subset = Subset(val_dataset, val_indices)
    # val_loader_subset = DataLoader(val_subset, batch_size=batch_size, shuffle=True)

