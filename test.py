import os
import torch
from tqdm import tqdm
import segmentation_models_pytorch as smp
from unet.config import *
from utils import metric
from unet.load_data import create_dataset_instances, create_dataloader
from utils.pred_result_visualize import show_worst_best_pred_result
from utils.get_today import get_today
import logging

DEVICE = torch.device('cuda:0')

def predict(test_loader):
    today = get_today()
    
    # log setting
    log = logging.getLogger()
    log.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s -  %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)

    if not os.path.exists('log/test/'+today):
        os.mkdir('log/test/'+today)

    file_handler = logging.FileHandler("log/test/"+today+"/"+today+"_test.log", mode='w')
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)

    # load checkpoint
    checkpoint = torch.load(CHECKPOINT_SAVE_PATH)

    model = smp.Unet(in_channels = 3, classes = 2)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device = DEVICE)

    # eval
    model.eval()
    test_roop =  tqdm(test_loader, desc='Test', leave=True)

    # initialize performance metrics
    y_pred_mask = []
    test_iou, test_acc, test_dice = 0, 0, 0

    metrics_dict = {}
    iou_scores, acc_scores, dice_scores = [], [], []

    with torch.no_grad():
        for i, data in enumerate(test_roop):
            image, mask = data
            image, mask = image.to(DEVICE), mask.to(DEVICE)

            output = model(image)

            mask_prob = output.softmax(dim = 1)
            mask_pred = mask_prob.argmax(dim = 1)
            y_pred_mask.append(mask_pred.detach().cpu())

            iou = metric.m_iou(output, mask, num_classes=2)
            iou_scores.append(iou)
            test_iou += iou

            acc = metric.pixel_accuracy(output, mask)
            acc_scores.append(acc)
            test_acc += acc

            dice = metric.DiceLoss()
            dice = dice.forward(output, mask)
            dice_scores.append(dice)
            test_dice += dice

    
    test_iou /= len(test_loader)
    test_acc /= len(test_loader)
    test_dice = 1 - (test_dice/len(test_loader))
    
    log.info('Test IoU: {:.3f}, Test Acc: {:.3f}, Test Dice: {:.3f}'.format(test_iou, test_acc, test_dice))
    y_pred_mask = torch.cat(y_pred_mask)

    # make metrics dict 
    iou_indexed_list = [(value, index) for index, value in enumerate(iou_scores)]
    iou_indexed_list.sort() 

    acc_indexed_list = [(value, index) for index, value in enumerate(acc_scores)]
    acc_indexed_list.sort()

    dice_indexed_list = [(value, index) for index, value in enumerate(dice_scores)]
    dice_indexed_list.sort()

    metrics_dict['iou'] = iou_indexed_list
    metrics_dict['acc'] = acc_indexed_list
    metrics_dict['dice'] = dice_indexed_list
    
    return y_pred_mask, metrics_dict


def test():
    # 1. Load data
    test_dataset = create_dataset_instances(mode='test')
    test_loader = create_dataloader(test_dataset)

    # 2. Predict
    y_pred_mask, metrics = predict(test_loader)

    # 3. Visualize Pred Result Image 
    show_worst_best_pred_result(test_dataset, y_pred_mask, metrics)

if __name__ == '__main__':
    test()