import matplotlib.pyplot as plt
from utils.get_today import get_today

# show worst and best pred result of specific metric
def show_worst_best_pred_result(dataset, pred_mask, metrics: dict, limit=3):
    """
    Plot a original image, mask overlay from the dataset and pred image 

    Args:
        dataset (Dataset): The custom BCSSDataset instance.
        pred_mask (list): predict mask 
        metrics (dict): consist of list (value, index) tuples. {'metric1':{(value1, index1), ...}, }
        limit (int): Number of images what you want to show. Default is 3. 
    """
    today = get_today()

    for metric in metrics.keys(): # iou, acc, dice
        metric_indexed_list = metrics.get(metric) # [(value1, index1), (value2, index2)...]
        worst_list = metric_indexed_list[:limit]
        best_list = metric_indexed_list[-limit:]

        # save images for each metrics up to a limit
        for i in range(limit):
            ## best image ## 
            best_idx = best_list[i][1]
            image, mask = dataset[best_idx]
            
            image_np = image.permute(1, 2, 0).numpy()  # Convert from PyTorch tensor to numpy array
            mask_np = mask.squeeze().numpy()  # Remove channel dimension and convert to numpy array
            pred_mask_np = pred_mask[best_idx].squeeze().numpy()

            plt.figure(figsize=(12, 6))
            plt.title('BEST '+metric+' '+str(best_list[i][0]))
            plt.gca().axes.xaxis.set_visible(False)
            plt.gca().axes.yaxis.set_visible(False)
            
            # Plot original image
            plt.subplot(1, 3, 1)
            plt.imshow(image_np)
            plt.title('Original Image')
            plt.axis('off')

            # Plot image with mask overlay
            plt.subplot(1, 3, 2)
            plt.imshow(image_np)
            plt.imshow(mask_np, alpha=0.5, vmin=0, vmax=1)  # Alpha controls the transparency
            plt.title('Image with Mask Overlay')
            plt.axis('off')

            # Plot image with pred mask overlay
            plt.subplot(1, 3, 3)
            plt.imshow(image_np)
            plt.imshow(pred_mask_np, alpha=0.5, vmin=0, vmax=1)  # Alpha controls the transparency
            plt.title('Image with Pred Mask Overlay')
            plt.axis('off')

            plt.show()

            plt.savefig('log/test/'+today+'/'+metric+'_best_'+str(best_idx)+'.png')

            ## worst image ##
            worst_idx = worst_list[i][1]
            image, mask = dataset[worst_idx]

            image_np = image.permute(1, 2, 0).numpy()  # Convert from PyTorch tensor to numpy array
            mask_np = mask.squeeze().numpy()  # Remove channel dimension and convert to numpy array
            pred_mask_np = pred_mask[worst_idx].squeeze().numpy()

            plt.figure(figsize=(12, 6))
            plt.title('WORST '+metric+' '+str(worst_list[i][0]))
            plt.gca().axes.xaxis.set_visible(False)
            plt.gca().axes.yaxis.set_visible(False)

            # Plot original image
            plt.subplot(1, 3, 1)
            plt.imshow(image_np)
            plt.title('Original Image')
            plt.axis('off')

            # Plot image with mask overlay
            plt.subplot(1, 3, 2)
            plt.imshow(image_np)
            plt.imshow(mask_np, alpha=0.5)  # Alpha controls the transparency
            plt.title('Image with Mask Overlay')
            plt.axis('off')

            # Plot image with pred mask overlay
            plt.subplot(1, 3, 3)
            plt.imshow(image_np)
            plt.imshow(pred_mask_np, alpha=0.5)  # Alpha controls the transparency
            plt.title('Image with Pred Mask Overlay')
            plt.axis('off')

            plt.show()

            plt.savefig('log/test/'+today+'/'+metric+'_worst_'+str(worst_idx)+'.png')
