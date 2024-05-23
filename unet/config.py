# SET CONFIG

## DATASET PATH ##
ROOT_PATH = 'data/BCSS_UNet/'

TRAIN_IMAGE_PATH = ROOT_PATH+'train'
VAL_IMAGE_PATH = ROOT_PATH+'val'
TEST_IMAGE_PATH = ROOT_PATH+'test'

TRAIN_MASK_PATH = ROOT_PATH+'train_mask'
VAL_MASK_PATH = ROOT_PATH+'val_mask'
TEST_MASK_PATH = ROOT_PATH+'test_mask'

CHECKPOINT_SAVE_PATH = 'model/last_checkpoint.pth.tar'

## PARAMETER ##
batch_size = 8

# Set the maximum learning rate for the optimizer
max_lr = 1e-3

# Define the number of epochs for training the model
num_epochs = 30

# Set the weight decay for regularization in the optimizer
weight_decay = 1e-4