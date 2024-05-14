

def predictions_mask(test_dataloader:torch.utils.data.DataLoader):
    
    checkpoint = torch.load("/kaggle/working/best_model.pth")

    loaded_model = smp.Unet(encoder_weights = None, classes = 21)

    loaded_model.load_state_dict(checkpoint)

    loaded_model.to(device = DEVICE)

    loaded_model.eval()

    y_pred_mask = []

    with torch.inference_mode():
        for batch,X in tqdm(enumerate(test_dataloader), total = len(test_dataloader)):
            X = X.to(device = DEVICE, dtype = torch.float32)
            mask_logit = loaded_model(X)
            mask_prob = mask_logit.softmax(dim = 1)
            mask_pred = mask_prob.argmax(dim = 1)
            y_pred_mask.append(mask_pred.detach().cpu())

    y_pred_mask = torch.cat(y_pred_mask)
    
    return y_pred_mask


if __name__ == '__main__':
    test()