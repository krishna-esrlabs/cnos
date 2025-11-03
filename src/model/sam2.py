import logging
from pathlib import Path
from typing import Union, List, Dict, Any, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import masks_to_boxes  # type: ignore

# SAM 2 imports
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


class SAM2Simple(object):
    """
    A minimal SAM2 wrapper in the same spirit as your MobileSAM class.
    - Expects RGB uint8 images
    - Optional width-based resize (keeps aspect)
    - Simple full-image point grid (no multi-crop, for simplicity)
    - Returns {"masks": Bool Tensor [N,H,W], "boxes": Float Tensor [N,4]} on device
    """

    pretrained_weight_dict = {
        "hiera_l": "sam2.1_hiera_large.pt",
        "hiera_b": "sam2.1_hiera_base.pt",
        "hiera_s": "sam2.1_hiera_small.pt",
        "hiera_t": "sam2.1_hiera_tiny.pt",
    }

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        model_type: str = "hiera_l",
        segmentor_width_size: Optional[int] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            checkpoint_dir: folder containing the SAM2 weights
            model_type: one of ["hiera_l","hiera_b","hiera_s","hiera_t"]
            segmentor_width_size: if set, resizes input image width to this value (keeps aspect)
            device: "cuda" or "cpu"
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.segmentor_width_size = segmentor_width_size
        self.current_device = device

        # Build SAM2
        ckpt = Path(checkpoint_dir) / self.pretrained_weight_dict[model_type]
        cfg = f"configs/sam2.1/sam2.1_{model_type}.yaml"
        logging.info(f"[SAM2Simple] Loading SAM2 {model_type} from {ckpt}")
        model = build_sam2(str(cfg), str(ckpt))
        model.to(self.device).eval()
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model,
            points_per_side=48,
            pred_iou_thresh=0.85,
            stability_score_thresh=0.95,
            stability_score_offset=0.8,
            mask_threshold=0.25,
            box_nms_thresh=0.6,
            crop_nms_thresh=0.7
        )
        logging.info("[SAM2Simple] Init done.")

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
