import numpy as np

def calculate_iou(predicted_mask, ground_truth_mask):
    intersection = np.logical_and(predicted_mask, ground_truth_mask)
    union = np.logical_or(predicted_mask, ground_truth_mask)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def compute_miou(predicted_masks, ground_truth_masks):
    predicted_masks=predicted_masks.cpu().detach().numpy()
    ground_truth_masks=ground_truth_masks.cpu().detach().numpy()
    batch_size = len(predicted_masks)
    num_classes = len(predicted_masks[0])
    iou_values = []

    for sample_idx in range(batch_size):
        sample_iou = []
        for class_idx in range(num_classes):
            iou = calculate_iou(predicted_masks[sample_idx][class_idx], ground_truth_masks[sample_idx][class_idx])
            sample_iou.append(iou)
        iou_values.append(sample_iou)

    iou_values = np.array(iou_values)
    class_mean_iou = np.mean(iou_values, axis=1)
    mIoU = np.sum(class_mean_iou)
    return mIoU
