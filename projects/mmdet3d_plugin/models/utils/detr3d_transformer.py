
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (MultiScaleDeformableAttention,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


@TRANSFORMER.register_module()
class Detr3DTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 decoder=None,
                 **kwargs):
        super(Detr3DTransformer, self).__init__(**kwargs)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.two_stage_num_proposals = two_stage_num_proposals
        self.reference_points = nn.Linear(self.embed_dims, 3)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention) or isinstance(m, Detr3DCrossAtten):
                m.init_weight()
        xavier_init(self.reference_points, distribution='uniform', bias=0.)

    def forward(self,
                mlvl_feats,
                query_embed,
                reg_branches=None,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats: list of tensors; shape=(B, 6, C, Hi, Wi)
            query_embed:  tensor(shape=(num_query, c))
            reg_branches: `nn.ModuleList`; each element is a MLP; len == num_layers
        """
        assert query_embed is not None
        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos).sigmoid()
        init_reference_out = reference_points
        # query_pos.shape=[B, num_query, emb_dim]
        # query.shape=[B, num_query, emb_dim]
        # reference_points.shape=[B, num_query, 3]

        query = query.permute(1, 0, 2)  # shape=[num_query, B, emb_dim]
        query_pos = query_pos.permute(1, 0, 2)  # shape=[num_query, B, emb_dim]

        # decoder
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=mlvl_feats,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            **kwargs)

        inter_references_out = inter_references
        # inter_states: tensor(shape=(num_layers, num_query, B, 256))
        # init_reference_out: tensor(shape=(B, num_query, 3))
        # inter_references_out: tensor(shape=(num_layers, B, num_query, 3))
        return inter_states, init_reference_out, inter_references_out


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class Detr3DTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(Detr3DTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

    def forward(self,
                query,
                reference_points=None,
                reg_branches=None,
                **kwargs):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query: shape=(num_query, B, emb_dim)`.
            reference_points: shape=(B, num_query, 3); within (0, 1)
            reg_branches: `nn.ModuleList`; each element is a MLP; len == num_layers
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            output = layer(
                output,
                reference_points=reference_points,
                **kwargs)  # (num_query, B, emb_dim)
            output = output.permute(1, 0, 2)  # (B, num_query, emb_dim)

            if reg_branches is not None:  # to generate new reference_points
                tmp = reg_branches[lid](output)  # (B, num_query, 10)
                
                assert reference_points.shape[-1] == 3

                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[
                    ..., :2] + inverse_sigmoid(reference_points[..., :2])
                new_reference_points[..., 2:3] = tmp[
                    ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])
                
                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()  # (B, num_query, 3)

            output = output.permute(1, 0, 2)  # (num_query, B, emb_dim)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


@ATTENTION.register_module()
class Detr3DCrossAtten(BaseModule):
    """An attention module used in Detr3d. 
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=5,
                 num_cams=6,
                 im2col_step=64,
                 pc_range=None,
                 dropout=0.1,
                 norm_cfg=None,
                 init_cfg=None,
                 batch_first=False):
        super(Detr3DCrossAtten, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_cams = num_cams
        self.attention_weights = nn.Linear(embed_dims,
                                           num_cams*num_levels*num_points)

        self.output_proj = nn.Linear(embed_dims, embed_dims)
      
        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.batch_first = batch_first

        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                reference_points=None,
                key_padding_mask=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value : list of tensor; multiple level features from FPN
                value[i].shape=[B, num_cams=6, emb_dim, Hi, Wi]
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            reference_points: shape=(B, num_query, 3); range (0, 1): top-left (0, 0), bottom-right(1, 1)
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        query = query.permute(1, 0, 2)  # (B, num_query, emb_dim)

        bs, num_query, _ = query.size()

        attention_weights = self.attention_weights(query).view(
            bs, 1, num_query, self.num_cams, self.num_points, self.num_levels)
        # attention_weights: shape=[B, 1, num_queries, num_cams, num_points, num_feature_levels]

        reference_points_3d, output, mask = feature_sampling(
            value, reference_points, self.pc_range, kwargs['img_metas'])
        # reference_points_3d: shape=(B, num_query, 3); reference_points.clone()
        # output: shape=(B, emd_dim, num_query, num_cams, num_points, num_levels);
        # mask: shape=(1, 1, 900, 6, 1, 1)
        output = torch.nan_to_num(output)
        mask = torch.nan_to_num(mask)

        attention_weights = attention_weights.sigmoid() * mask  # shape same as above
        output = output * attention_weights
        output = output.sum(-1).sum(-1).sum(-1)
        output = output.permute(2, 0, 1)  # (num_query, B, emd_dim)
        
        output = self.output_proj(output)  # (num_query, B, emd_dim)
        pos_feat = self.position_encoder(inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)
        # pos_feat: shape=(num_query, B, emd_dim)

        return self.dropout(output) + inp_residual + pos_feat


def feature_sampling(mlvl_feats, reference_points, pc_range, img_metas):
    lidar2img = []
    for img_meta in img_metas:
        lidar2img.append(img_meta['lidar2img'])
    lidar2img = np.asarray(lidar2img)
    lidar2img = reference_points.new_tensor(lidar2img) # (B, N, 4, 4)
    reference_points = reference_points.clone()
    reference_points_3d = reference_points.clone()
    reference_points[..., 0:1] = reference_points[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]
    # reference_points (B, num_queries, 4)
    reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
    B, num_query = reference_points.size()[:2]
    num_cam = lidar2img.size(1)
    reference_points = reference_points.view(B, 1, num_query, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)
    lidar2img = lidar2img.view(B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)
    reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)
    eps = 1e-5
    mask = (reference_points_cam[..., 2:3] > eps)
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps)
    reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
    reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]
    reference_points_cam = (reference_points_cam - 0.5) * 2
    mask = (mask & (reference_points_cam[..., 0:1] > -1.0) 
                 & (reference_points_cam[..., 0:1] < 1.0) 
                 & (reference_points_cam[..., 1:2] > -1.0) 
                 & (reference_points_cam[..., 1:2] < 1.0))
    mask = mask.view(B, num_cam, 1, num_query, 1, 1).permute(0, 2, 3, 1, 4, 5)
    mask = torch.nan_to_num(mask)
    sampled_feats = []
    for lvl, feat in enumerate(mlvl_feats):
        B, N, C, H, W = feat.size()
        feat = feat.view(B*N, C, H, W)
        reference_points_cam_lvl = reference_points_cam.view(B*N, num_query, 1, 2)
        sampled_feat = F.grid_sample(feat, reference_points_cam_lvl)
        sampled_feat = sampled_feat.view(B, N, C, num_query, 1).permute(0, 2, 3, 1, 4)
        sampled_feats.append(sampled_feat)
    sampled_feats = torch.stack(sampled_feats, -1)
    sampled_feats = sampled_feats.view(B, C, num_query, num_cam,  1, len(mlvl_feats))
    return reference_points_3d, sampled_feats, mask
