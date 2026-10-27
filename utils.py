import cv2
import numpy as np


def save_on_batch(images, masks, pred, names, vis_path):
    for i in range(pred.shape[0]):
        pred_tmp = pred[i][0].cpu().detach().numpy()
        mask_tmp = masks[i].cpu().detach().numpy()

        pred_tmp = ((pred_tmp >= 0.5) * 255).astype(np.uint8)
        mask_tmp = ((mask_tmp > 0) * 255).astype(np.uint8)

        cv2.imwrite(vis_path + names[i][:-4] + "_pred.jpg", pred_tmp)
        cv2.imwrite(vis_path + names[i][:-4] + "_gt.jpg", mask_tmp)
