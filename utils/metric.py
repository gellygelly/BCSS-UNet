import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def pixel_accuracy(pred_mask, gt_mask):
    """
    Calculate the pixel-wise accuracy of classification predictions.

    This function computes the accuracy by comparing the predicted classes
    (obtained by applying softmax followed by argmax on the model's output)
    to the ground truth classes provided in the mask.

    Args:
        pred_mask (torch.Tensor): The raw output (logits) from the neural network.
                               Shape should be (batch_size, num_classes, height, width).
        mask (torch.Tensor): The ground truth labels for each pixel.
                             Shape should be (batch_size, height, width) with integer values representing classes.

    Returns:
        float: The computed pixel-wise accuracy as a proportion of correct predictions.
    """
    with torch.no_grad():  # Ensure no gradients are calculated
        # Apply softmax to the output to convert to probability distributions
        # Then, use argmax to get the most likely class prediction for each pixel
        predictions = torch.argmax(F.softmax(pred_mask, dim=1), dim=1)

        # Compare the predictions with the true labels (mask)
        # Create a binary tensor where 'True' indicates correct prediction
        correct_predictions = (predictions == gt_mask)

        # Calculate accuracy
        # Sum up the correct predictions and divide by the total number of predictions
        accuracy = torch.sum(correct_predictions).item() / correct_predictions.numel()

    return accuracy

def m_iou(pred_mask, gt_mask, smooth=1e-10, num_classes=2):
    """
    Calculate the mean Intersection over Union (mIoU) for predicted segmentation masks.

    mIoU is a common metric for evaluating the performance of a segmentation model. It computes
    the IoU for each class and then averages them. This function handles multi-class predictions
    and can deal with cases where a class is not present in the ground truth.

    Args:
        pred_mask (torch.Tensor): The predicted segmentation mask. Shape should be (batch_size, num_classes, height, width).
        mask (torch.Tensor): The ground truth segmentation mask. Shape should be (batch_size, height, width).
        smooth (float, optional): A small value to avoid division by zero. Default is 1e-10.
        num_classes (int, optional): The number of classes in the segmentation task. Default is 3.

    Returns:
        float: The computed mean IoU across all classes.
    """
    with torch.no_grad():  # No gradients needed for metric calculation
        # Apply softmax to convert to probability distribution, then use argmax to get class prediction
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        gt_mask = gt_mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, num_classes):  # Loop through each class
            true_class = pred_mask == clas
            true_label = gt_mask == clas

            # Handle cases where the class is not present in the labels
            if true_label.long().sum().item() == 0:
                iou_per_class.append(np.nan)
            else:
                # Calculate intersection and union for the current class
                intersect = torch.logical_and(true_class, true_label).sum().item()
                union = torch.logical_or(true_class, true_label).sum().item()

                # Calculate IoU for the current class
                iou = (intersect + smooth) / (union + smooth)
                iou_per_class.append(iou)

        # Compute the mean IoU, ignoring NaN values
        return np.nanmean(iou_per_class)

class DiceLoss(nn.Module):
    """
    Dice Loss class for semantic segmentation.

    Dice Loss is a common loss function used for image segmentation, especially when dealing with 
    highly imbalanced datasets. It measures the overlap between two samples and is defined as 
    2 * |X∩Y| / (|X|+|Y|), where |X| and |Y| are the cardinalities of two sets.

    This implementation of Dice Loss supports multi-class segmentation tasks.
    """

    def __init__(self):
        """
        Initializes the DiceLoss instance.
        """
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, eps=1e-6):
        """
        Forward pass for the Dice Loss calculation.

        Args:
            inputs (torch.Tensor): The predicted logits from the model.
                                   Shape: (batch_size, num_classes, height, width)
            targets (torch.Tensor): The ground truth segmentation masks.
                                    Shape: (batch_size, height, width)
            eps (float): A small value to avoid division by zero in the loss calculation.
                         Default value is 1e-6.

        Returns:
            torch.Tensor: The computed Dice Loss.
        """
        # Apply softmax to the inputs to convert logits to probabilities
        inputs = torch.softmax(inputs, dim=1)

        # Create the one-hot encoding of targets for multi-class compatibility
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1])
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # Reshape to match input tensor shape

        # Flatten the inputs and targets for element-wise operations
        inputs_flat = inputs.view(inputs.shape[0], inputs.shape[1], -1)
        targets_flat = targets_one_hot.view(targets_one_hot.shape[0], targets_one_hot.shape[1], -1)

        # Calculate intersection and union for each class
        intersection = torch.sum(inputs_flat * targets_flat, dim=2)
        total = torch.sum(inputs_flat, dim=2) + torch.sum(targets_flat, dim=2)

        # Calculate dice score for each class and then average across classes
        dice = (2. * intersection + eps) / (total + eps)  # Add eps for numerical stability
        dice = dice.mean(dim=1)  # Average over classes

        # Dice loss is 1 minus the dice score
        return 1.0 - dice.mean()