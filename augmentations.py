import torch
import torch.nn as nn
import kornia.augmentation as K


class KorniaAugmentation(nn.Module):
    def __init__(self, p_hflip=0.5, rotation_degrees=15):
        super().__init__()
        self.aug = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=p_hflip, keepdim=True),
            K.RandomRotation(degrees=rotation_degrees, keepdim=True),
            data_keys=["input", "keypoints"],
            keepdim=True,
        )
    
    def forward(self, images, keypoints):
        if keypoints.dim() == 2:
            keypoints = keypoints.unsqueeze(1)
        
        aug_images, aug_keypoints = self.aug(images, keypoints)
        
        if aug_keypoints.dim() == 3 and aug_keypoints.size(1) == 1:
            aug_keypoints = aug_keypoints.squeeze(1)
        
        return aug_images, aug_keypoints

