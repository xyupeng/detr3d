# CUDA_VISIBLE_DEVICES=7 python ./tools/train.py projects/configs/detr3d_waymo_debug.py --cfg-options data.samples_per_gpu=2
# CUDA_VISIBLE_DEVICES=4,5,6,7 ./tools/dist_train.sh projects/configs/detr3d_waymo_debug.py 8

# 0. base & custom_imports
_base_ = [
    '../../configs/mmdet3d/_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['projects.mmdet3d_plugin'],
    allow_failed_imports=False)

# required by model
point_cloud_range = [-74.88, -74.88, -2, 74.88, 74.88, 4]
voxel_size = [0.32, 0.32, 6]

input_modality = dict(use_lidar=False, use_camera=True)
cams = ('CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT')

class_names = ('Car', 'Pedestrian', 'Cyclist')
num_classes = len(class_names)

# 1. model
model = dict(
    type='Detr3D',
    use_grid_mask=True,
    img_backbone=dict(
        type='ResNet',
        depth=50,  # default 101; 50 is ok
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)
    ),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True
    ),
    pts_bbox_head=dict(
        type='Detr3DHead',
        num_query=900,
        num_classes=num_classes,
        in_channels=256,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        code_size=8,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        # (cx, cy, w, l, cz, h, rot.sin(), rot.cos()); if code_size == 10, append (vx, vy)
        transformer=dict(
            type='Detr3DTransformer',  # mmdet3d_plugin/models/utils/detr3d_transformer.py
            decoder=dict(
                type='Detr3DTransformerDecoder',  # mmdet3d_plugin/models/utils/detr3d_transformer.py
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',  # mmdet/models/utils/transformer.py
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',  # mmcv/cnn/bricks/transformer.py
                            embed_dims=256,
                            num_heads=8,  # head_dims == 256 // 8 == 32
                            dropout=0.1),
                        dict(
                            type='Detr3DCrossAtten',  # mmdet3d_plugin/models/utils/detr3d_transformer.py
                            pc_range=point_cloud_range,
                            num_points=1,
                            num_cams=len(cams),
                            embed_dims=256)
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')
                )
            )
        ),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-84.8, -84.8, -10.0, 84.8, 84.8, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=num_classes
        ),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
                pc_range=point_cloud_range
            )
        )
    )
)

# 2. dataset
dataset_type = 'WaymoMultiViewDataset'
data_root = './data/waymo/kitti_format/'
file_client_args = dict(backend='disk')
ann_file_train = 'waymo_multi_view_infos_train.pkl'
ann_file_val = 'waymo_multi_view_infos_val.pkl'
ann_file_test = 'waymo_multi_view_infos_val.pkl'

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675],
    std=[1.0, 1.0, 1.0],
    to_rgb=False,
)

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
]
'''
    TODO:
    'img_metas': dict
    'gt_bboxes_3d': LiDARInstance3DBoxes, FloatTensor(shape=[num_gt_bboxes, 7]) 
    'gt_labels_3d': LongTensor(shape=[num_gt_bboxes]) 
    'img': shape=(6, 3, 928, 1600); within [-255, 255] (see img_norm_cfg.mean)
'''

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=6,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + ann_file_train,
        split='training',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        cams=cams,
        test_mode=False,
        box_type_3d='LiDAR',
        load_interval=5,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + ann_file_val,
        split='training',
        pipeline=test_pipeline,
        modality=input_modality,
        cams=cams,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + ann_file_test,
        split='training',
        pipeline=test_pipeline,
        modality=input_modality,
        cams=cams,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
    )
)
evaluation = dict(interval=24, pipeline=test_pipeline)  # interval=2
# eval_hook priority='LOW'; after save_checkpoint() whose priority='NORMAL'

# 3. schedule
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
epochs = 24
runner = dict(type='EpochBasedRunner', max_epochs=epochs)

# 4. runtime
checkpoint_config = dict(interval=6)
log_config = dict(interval=100)
load_from = None
resume_from = None
workflow = [('train', epochs), ('val', 1)]
