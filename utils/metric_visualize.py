import matplotlib.pyplot as plt

def show_loss_graph(history):
    # Plotting the training and validation losses
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting the learning rate
    plt.subplot(1, 2, 2)
    plt.plot(history['lrs'], label='Learning Rate')
    plt.title('Learning Rate Curve')
    plt.xlabel('Steps')
    plt.ylabel('Learning Rate')
    plt.legend()

    plt.show()

    plt.savefig('log/train/loss_graph.png')

def show_acc_graph(history):
    # Plotting Training and Validation Accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plotting Training and Validation mIoU
    plt.subplot(1, 2, 2)
    plt.plot(history['train_miou'], label='Train mIoU')
    plt.plot(history['val_miou'], label='Validation mIoU')
    plt.title('Training and Validation mIoU')
    plt.xlabel('Epochs')
    plt.ylabel('mIoU')
    plt.legend()

    plt.show()

    plt.savefig('log/train/acc_graph.png')