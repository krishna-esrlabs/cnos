import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl
import logging
import numpy as np
from PIL import Image
from src.utils.bbox_utils import CropResizePad, CustomResizeLongestSide
from torchvision.utils import make_grid, save_image
from src.model.utils import BatchedData
from copy import deepcopy

from transformers import AutoProcessor, AutoModel

descriptor_size = {
    "google/siglip2-base-patch16-224": 768,
    "google/siglip2-base-patch16-256": 768,
    "google/siglip2-base-patch16-384": 768,
    "google/siglip2-base-patch16-512": 768,
    "google/siglip2-base-patch16-naflex": 768,
    "google/siglip2-base-patch32-256": 768,

    "google/siglip2-large-patch16-256": 1024,
    "google/siglip2-large-patch16-384": 1024,
    "google/siglip2-large-patch16-512": 1024,

    "google/siglip2-giant-opt-patch16-256": 1536,
    "google/siglip2-giant-opt-patch16-384": 1536,

    "google/siglip2-so400m-patch14-224": 1152,
    "google/siglip2-so400m-patch14-384": 1152,
    "google/siglip2-so400m-patch16-256": 1152,
    "google/siglip2-so400m-patch16-384": 1152,
    "google/siglip2-so400m-patch16-512": 1152,
    "google/siglip2-so400m-patch16-naflex": 1152
}


class CustomSIGLIPv2(pl.LightningModule):
    def __init__(
        self,
        model_name,
        model,
        token_name,
        image_size,
        chunk_size,
        descriptor_width_size,
        patch_size=14,
    ):
        super().__init__()
        self.model_name = model_name
        self.model = model
        print(model)
        self.token_name = token_name
        self.chunk_size = chunk_size
        self.patch_size = patch_size
        self.proposal_size = image_size
        self.descriptor_width_size = descriptor_width_size
        logging.info(f"Init CustomSIGLIPv2 done!")
        self.preprocessor = AutoProcessor.from_pretrained(f"{self.model_name}")
        # use for global feature
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.rgb_normalize = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=self.mean, std=self.std),
            ]
        )
        self.inv_rgb_normalize = T.Compose(
            [
                T.Normalize(mean=[-m/s for m, s in zip(self.mean, self.std)],
                            std=[1/s for s in self.std]),
                T.Lambda(lambda x: torch.clamp(x, 0, 1)),
                T.ToPILImage()
            ]
        )
        self.rgb_proposal_processor = CropResizePad(self.proposal_size)
        self.rgb_resize = CustomResizeLongestSide(
            descriptor_width_size, dividable_size=self.patch_size
        )
        logging.info(
            f"Init CustomSIGLIPv2 with full size={descriptor_width_size} and proposal size={self.proposal_size} done!"
        )

    def process_rgb_proposals(self, image_np, masks, boxes):
        """
        1. Normalize image with SIGLIPv2 transfom
        2. Mask and crop each proposals
        3. Resize each proposals to predefined longest image size
        """
        num_proposals = len(masks)
        rgb = self.rgb_normalize(image_np).to(masks.device).float()
        rgbs = rgb.unsqueeze(0).repeat(num_proposals, 1, 1, 1)
        masked_rgbs = rgbs * masks.unsqueeze(1)
        processed_masked_rgbs = self.rgb_proposal_processor(
            masked_rgbs, boxes
        )  # [N, 3, target_size, target_size]
        return processed_masked_rgbs

    @torch.no_grad()
    def compute_features(self, images, token_name):
        if token_name == "x_norm_clstoken":
            if images.shape[0] > self.chunk_size:
                features = self.forward_by_chunk(images)
            else:
                inputs = self.preprocessor(
                    images=[self.inv_rgb_normalize(img) for img in images],
                    return_tensors = "pt")
                inputs = inputs.to(self.model.device)
                features = self.model.get_image_features(**inputs)
        else:  # get both features
            raise NotImplementedError
        return features

    @torch.no_grad()
    def forward_by_chunk(self, processed_rgbs):
        batch_rgbs = BatchedData(batch_size=self.chunk_size, data=processed_rgbs)
        del processed_rgbs  # free memory
        features = BatchedData(batch_size=self.chunk_size)
        for idx_batch in range(len(batch_rgbs)):
            feats = self.compute_features(
                batch_rgbs[idx_batch], token_name="x_norm_clstoken"
            )
            features.cat(feats)
        return features.data

    @torch.no_grad()
    def forward_cls_token(self, image_np, proposals):
        processed_rgbs = self.process_rgb_proposals(
            image_np, proposals.masks, proposals.boxes
        )
        return self.forward_by_chunk(processed_rgbs)

    @torch.no_grad()
    def forward(self, image_np, proposals):
        return self.forward_cls_token(image_np, proposals)
