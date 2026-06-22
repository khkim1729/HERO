from __future__ import annotations

from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ScaleIntensityRangePercentilesd,
    SpatialPadd,
    Spacingd,
)


def get_train_transforms():
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(4.0, 4.0, 6.4), mode=("bilinear", "nearest")),
        CenterSpatialCropd(keys=["image", "label"], roi_size=(64, 64, 64)),
        SpatialPadd(keys=["image", "label"], spatial_size=(64, 64, 64)),
        ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1),
    ])
    return train_transforms


def get_eval_transforms():
    return get_train_transforms()

