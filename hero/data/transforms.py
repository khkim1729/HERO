from __future__ import annotations

from monai.transforms import (
    CropForegroundd,
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


def _base_transforms(pixdim: tuple[float, float, float]):
    return [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
        ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1),
        CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
    ]


def get_train_transforms(pixdim: tuple[float, float, float], roi_size: tuple[int, int, int]):
    return Compose(_base_transforms(pixdim) + [
        CenterSpatialCropd(keys=["image", "label"], roi_size=roi_size),
        SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
    ])


def get_eval_transforms(pixdim: tuple[float, float, float], roi_size: tuple[int, int, int]):
    return Compose(_base_transforms(pixdim) + [
        CenterSpatialCropd(keys=["image", "label"], roi_size=roi_size),
        SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
    ])


def get_infer_transforms(pixdim: tuple[float, float, float], roi_size: tuple[int, int, int]):
    return Compose(_base_transforms(pixdim) + [
        SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
    ])
