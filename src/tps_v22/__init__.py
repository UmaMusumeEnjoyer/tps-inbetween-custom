# -*- coding: utf-8 -*-
"""
Package pipelines.tps_v22:
Pipeline Tinh chỉnh & Huấn luyện TPS-Inbetween V2.2
Giải quyết 4 vấn đề cốt lõi của V2.0.2:
1. Hiện tượng phình nét (Line Thickening - Density Ratio 1.31)
2. Mất cân xứng chiều thời gian (t=0.20 vs t=0.80)
3. Độ vênh metric giữa validation và fast benchmark
4. Tối ưu độ trễ suy luận (Latency) và chuẩn hóa đồng bộ CUDA
"""

from .losses import (
    StrokeDensityRegularizationLoss,
    DirectionalTangentCoherenceLoss,
    SoftClDiceLoss,
    CompositeLossEngine_v22
)
from .data_pipeline import (
    TemporalReversalAugmentation,
    SakugaDynamic5050Sampler,
    SakugaBoundaryMotionWeightedSampler,
    compute_boundary_motion_weights,
    SakugaGoldenDataset_v22
)
from .smart_checkpoint_manager import SmartCheckpointManager_v22
from .metrics import calculate_standardized_metrics
from .optimized_engine import TPSInbetweenFinetune_v22

__all__ = [
    'StrokeDensityRegularizationLoss',
    'DirectionalTangentCoherenceLoss',
    'SoftClDiceLoss',
    'CompositeLossEngine_v22',
    'TemporalReversalAugmentation',
    'SakugaDynamic5050Sampler',
    'SakugaBoundaryMotionWeightedSampler',
    'compute_boundary_motion_weights',
    'SakugaGoldenDataset_v22',
    'SmartCheckpointManager_v22',
    'calculate_standardized_metrics',
    'TPSInbetweenFinetune_v22'
]
