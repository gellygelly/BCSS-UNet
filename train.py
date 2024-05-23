import time
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import numpy as np
from unet.config import *
from unet.load_data import create_dataset_instances, create_dataloader
from utils import metric
from utils.metric_visualize import show_loss_graph, show_acc_graph
from utils.get_today import get_today
import logging

DEVICE = torch.device('cuda:0')

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """
    Save the current training state as a checkpoint.

    Args:
        state (dict): State to be saved, including model and optimizer state, training history, etc.
        filename (str): Filename for saving the checkpoint.
    """
    
    torch.save(state, filename)

def get_lr(optimizer):
    """
    Retrieve the current learning rate from the optimizer.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer being used in training.

    Returns:
        float: The current learning rate.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit(epochs, model, train_loader, val_loader, criterion1, criterion2, optimizer, scheduler, patch=False):
    """
    Train and validate the model over a specified number of epochs.

    Args:
        epochs (int): Number of epochs to train the model.
        model (torch.nn.Module): The neural network model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        criterion1 (torch.nn.Module): The primary loss function.
        criterion2 (torch.nn.Module): The secondary loss function.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        patch (bool): If True, applies any additional modifications or fixes during training.

    Returns:
        dict: A dictionary containing the training history (losses, accuracies, IoU scores, learning rates).
    """
    today = get_today()
    
    # log setting
    log = logging.getLogger()
    log.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s -  %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)

    if not os.path.exists('log/train/'+today):
        os.mkdir('log/train/'+today)

    file_handler = logging.FileHandler("log/train/"+today+"/train.log", mode='w')
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)

    torch.cuda.empty_cache()
    # Initialize tracking variables for performance metrics
    train_losses, test_losses, val_iou, val_acc, train_iou, train_acc, lrs = [], [], [], [], [], [], []
    min_loss, max_iou = np.inf, 0
    decrease, not_improve = 1, 0

    model.to(DEVICE)
    fit_time = time.time()
    
    # Load checkpoint if it exists
    if os.path.exists(CHECKPOINT_SAVE_PATH):
        checkpoint = torch.load(CHECKPOINT_SAVE_PATH)
        # Restore states from the checkpoint
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        train_losses, test_losses, val_iou, val_acc, train_iou, train_acc, lrs = (
            checkpoint['train_losses'], checkpoint['test_losses'],
            checkpoint['val_iou'], checkpoint['val_acc'],
            checkpoint['train_iou'], checkpoint['train_acc'],
            checkpoint['lrs']
        )
        decrease, not_improve, min_loss, max_iou = (
            checkpoint['decrease'], checkpoint['not_improve'],
            checkpoint['min_loss'], checkpoint['max_iou']
        )
        start_epoch = checkpoint['epoch']
        
        log.info(f'Continue training from {start_epoch} epoch!')
    else:
        start_epoch = 0
    

    for e in range(start_epoch, epochs):     
        since = time.time()
        running_loss, iou_score, accuracy = 0, 0, 0

        # Training loop
        model.train()
        train_loop = tqdm(train_loader, desc='Training', leave=True)
        for i, data in enumerate(train_loop):
            image, mask = data
            image, mask = image.to(DEVICE), mask.to(DEVICE)

            # Forward pass
            output = model(image)

            # Compute loss and update weights
            loss = criterion1(output, mask) + criterion2(output, mask)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Update metrics
            iou_score += metric.m_iou(output, mask)
            accuracy += metric.pixel_accuracy(output, mask)
            running_loss += loss.item()

            # Update tqdm loop with current metrics
            train_loop.set_postfix(loss=(running_loss / (i + 1)),
                                   mIoU=iou_score / (i + 1),
                                   acc=accuracy / (i + 1))

            # Step learning rate scheduler
            lrs.append(get_lr(optimizer))
            scheduler.step()

        # Validation loop
        model.eval()
        test_loss, test_accuracy, val_iou_score = 0, 0, 0
        val_loop = tqdm(val_loader, desc='Validation', leave=True)
        with torch.no_grad():
            for i, data in enumerate(val_loop):
                image, mask = data
                image, mask = image.to(DEVICE), mask.to(DEVICE)

                # Forward pass and loss computation
                output = model(image)
                loss = criterion1(output, mask) + criterion2(output, mask)

                # Update metrics
                val_iou_score += metric.m_iou(output, mask)
                test_accuracy += metric.pixel_accuracy(output, mask)
                test_loss += loss.item()

                # Update tqdm loop with current metrics
                val_loop.set_postfix(loss=test_loss / (i + 1),
                                     mIoU=val_iou_score / (i + 1),
                                     acc=test_accuracy / (i + 1))

        # Calculate mean metrics for the epoch
        train_losses.append(running_loss / len(train_loader))
        test_losses.append(test_loss / len(val_loader))

        # Checkpointing and early stopping logic
        if min_loss > (test_loss/len(val_loader)):
                log.info('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, (test_loss/len(val_loader))))
                min_loss = (test_loss/len(val_loader))
                decrease += 1
                torch.save(model, 'Unet-mIoU-best.pt'.format(val_iou_score/len(val_loader)))
                    

        if (test_loss/len(val_loader)) > min_loss:
            not_improve += 1
            min_loss = (test_loss/len(val_loader))
            log.info(f'Loss Not Decrease for {not_improve} time')
            if not_improve == 5:
                log.info('Loss not decrease for 5 times, Stop Training')
                break
        else:
            not_improve = 0

        # Update history after each epoch
        val_iou.append(val_iou_score / len(val_loader))
        val_acc.append(test_accuracy / len(val_loader))
        train_iou.append(iou_score / len(train_loader))
        train_acc.append(accuracy / len(train_loader))

        # Print epoch summary
        log.info("Epoch:{}/{}.. Train Loss: {:.3f}, Val Loss: {:.3f}, Train mIoU: {:.3f}, Val mIoU: {:.3f}, Train Acc: {:.3f}, Val Acc: {:.3f}, Time: {:.2f}m".format(
            e + 1, epochs,
            running_loss / len(train_loader),
            test_loss / len(val_loader),
            iou_score / len(train_loader),
            val_iou_score / len(val_loader),
            accuracy / len(train_loader),
            test_accuracy / len(val_loader),
            (time.time() - since) / 60))
        
        # Save checkpoint
        state = {
            'epoch': e + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'train_losses': train_losses,
            'test_losses': test_losses,
            'val_iou': val_iou,
            'val_acc': val_acc,
            'train_iou': train_iou,
            'train_acc': train_acc,
            'lrs': lrs,
            'decrease': decrease,
            'not_improve': not_improve,
            'min_loss': min_loss,
            'max_iou': max_iou
        }
        save_checkpoint(state, filename='model/last_checkpoint.pth.tar')

    # Compile history of training
    history = {
        'train_loss': train_losses, 'val_loss': test_losses,
        'train_miou': train_iou, 'val_miou': val_iou,
        'train_acc': train_acc, 'val_acc': val_acc,
        'lrs': lrs
    }
    log.info('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))

    return history


def train():
    # 1. Load data
    train_dataset = create_dataset_instances(mode='train')
    train_loader = create_dataloader(train_dataset)

    val_dataset = create_dataset_instances(mode='val')
    val_loader = create_dataloader(val_dataset)

    # 2. Define model
    model = smp.Unet(in_channels = 3, classes = 2) # background(0), tumor(1)

    # Define the primary loss function with label smoothing to improve generalization
    # Label smoothing helps to make the model less confident on the training data
    criterion1 = nn.CrossEntropyLoss(label_smoothing=0.1).to(DEVICE)

    # Define the secondary loss function, Dice Loss, useful for handling class imbalance in segmentation tasks
    criterion2 = metric.DiceLoss().to(DEVICE)

    # Initialize the optimizer with weight decay for regularization
    # AdamW is an optimizer with an adaptive learning rate and weight decay fix
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)

    # Define a learning rate scheduler
    # OneCycleLR adjusts the learning rate during training for better convergence
    # It starts with a lower LR, increases it, and then decreases it towards the end
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=num_epochs,
                                                    steps_per_epoch=len(train_loader))

    # 3. Train
    # The 'fit' function trains the model over the specified number of epochs
    # and returns a history of training metrics like loss and accuracy
    history = fit(num_epochs, model, train_loader, val_loader, criterion1, criterion2, optimizer, scheduler)

    # 4. Visualize_graph
    show_loss_graph(history)
    show_acc_graph(history)


if __name__ == '__main__':
    train()