import logging
from pathlib import Path
from typing import Union, List, Dict, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator


class MobileSAM(object):
    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        stability_score_thresh: float = 0.95,
        segmentor_width_size=None,
        device=None
    ):
        model_type = "vit_t"
        self.segmentor_width_size = segmentor_width_size
        self.current_device = device
        mobile_sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        mobile_sam.to(device=self.current_device)
        mobile_sam.eval()
        self.mask_generator = SamAutomaticMaskGenerator(mobile_sam)
        # TODO: currently unused threshold
        self.stability_score_thresh = stability_score_thresh
        logging.info(f"Init MobileSam done!")

    def postprocess_resize(self, detections, orig_size, update_boxes=False):
        detections["masks"] = F.interpolate(
            detections["masks"].unsqueeze(1).float(),
            size=(orig_size[0], orig_size[1]),
            mode="bilinear",
            align_corners=False,
        )[:, 0, :, :]
        if update_boxes:
            scale = orig_size[1] / self.segmentor_width_size
            detections["boxes"] = detections["boxes"].float() * scale
            detections["boxes"][:, [0, 2]] = torch.clamp(
                detections["boxes"][:, [0, 2]], 0, orig_size[1] - 1
            )
            detections["boxes"][:, [1, 3]] = torch.clamp(
                detections["boxes"][:, [1, 3]], 0, orig_size[0] - 1
            )
        return detections

    @torch.no_grad()
    def generate_masks(self, image) -> List[Dict[str, Any]]:
        if self.segmentor_width_size is not None:
            orig_size = image.shape[:2]
            h, w = image.shape[:2]
            target_h = int(h * (self.segmentor_width_size / w))
            image = cv2.resize(image,
                               (self.segmentor_width_size, target_h),
                               interpolation=cv2.INTER_NEAREST)

        # dict_keys(['segmentation', 'area', 'bbox', 'predicted_iou',
        # 'point_coords', 'stability_score', 'crop_box'])
        detections = self.mask_generator.generate(image)

        masks = []
        bboxes = []
        for i in range(len(detections)):
            masks.append(detections[i]['segmentation'])
            x, y, w, h = detections[i]['bbox']
            bboxes.append([x, y, x+w, y+h])

        masks = torch.from_numpy(np.array(masks))
        bboxes = torch.from_numpy(np.array(bboxes))

        mask_data = {
            "masks": masks.to(self.current_device),
            "boxes": bboxes.to(self.current_device),
        }
        if self.segmentor_width_size is not None:
            mask_data = self.postprocess_resize(mask_data, orig_size, True)
        return mask_data
