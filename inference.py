# TODO 이미지 한 장 or 폴더 내 이미지를 대상으로 inference 하는 코드 작성 예정

def test_unet():
    """
    Test function for the UNet model to verify input-output compatibility.
    """
    n_channels = 3  # Number of input channels (e.g., RGB has 3 channels)
    n_classes = 1   # Number of output classes for segmentation
    model = UNet(n_channels, n_classes)

    # Create a dummy input tensor (batch size, channels, height, width)
    # Adjust the size (512, 512) as needed for your specific use case
    dummy_input = torch.randn(1, n_channels, 512, 512)

    # TODO 위 부분 이미지 한 장읽어서 im

    # Forward pass through the model
    output = model(dummy_input)

    # Check the output shape
    assert output.shape == (1, n_classes, 512, 512), f"Output shape is {output.shape}, but expected (1, {n_classes}, 512, 512)"

    print("UNet model test passed. Output shape is correct.")

# Run the test function
test_unet()

# Select GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(num_channels=3, num_classes=1)
model.to(device)


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