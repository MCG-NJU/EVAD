import math
import random
import copy
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import sigmoid_focal_loss_jit
from timm.models.layers import trunc_normal_
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes

from slowfast.models.losses import focal_loss_wo_logits_jit
from slowfast.datasets.cv2_transform import clip_boxes_tensor
from .vit_model import ContextRefinementDecoder, get_sinusoid_encoding_table, interpolate_pos_embed_online

_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)


class EVADRoIHead(nn.Module):
    """ EVAD RoI head.
    """

    def __init__(self, cfg, dim_in):
        """
        EVADRoIHead takes p pathways as input where p = 1.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
            dim_in (list): the list of channel dimensions of the p inputs to the
                RoIHead.
        """
        super(EVADRoIHead, self).__init__()
        self.cfg = cfg
        self.use_fpn = cfg.MODEL.SparseRCNN.USE_FPN
        self.gt_boxes_prob = cfg.MODEL.SparseRCNN.GT_BOXES_PROB
        self.num_pathways = len(dim_in)
        self.device = torch.device(cfg.MODEL.DEVICE)

        pool_size = [[cfg.DATA.NUM_FRAMES // cfg.ViT.TUBELET_SIZE, 1, 1]]
        resolution = [[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2]
        scale_factor = [cfg.DETECTION.SPATIAL_SCALE_FACTOR]

        for pathway in range(self.num_pathways):
            spatial_pool = nn.AdaptiveMaxPool3d(
                [pool_size[pathway][0], 1, 1],
            )
            self.add_module("s{}_spool".format(pathway), spatial_pool)
            temporal_pool = nn.AvgPool3d(
                [pool_size[pathway][0], 1, 1], stride=1
            )
            self.add_module("s{}_tpool".format(pathway), temporal_pool)

            pooler = ROIPooler(
                output_size=resolution[pathway],
                scales=[1.0 / scale_factor[pathway]],
                sampling_ratio=cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO,
                pooler_type=cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
            )
            self.add_module("s{}_roi".format(pathway), pooler)
            if self.use_fpn:
                keyframe_pooler = self._init_box_pooler(cfg)
                self.add_module("s{}_keyroi".format(pathway), keyframe_pooler)

            if pathway == 0:
                rcnn_head = RCNNHead(cfg)
                head_series = _get_clones(rcnn_head, cfg.MODEL.SparseRCNN.NUM_HEADS)
                self.add_module("s{}_headseries".format(pathway), head_series)

        self.return_intermediate = cfg.MODEL.SparseRCNN.DEEP_SUPERVISION
        self.use_focal = cfg.MODEL.SparseRCNN.USE_FOCAL
        self.num_classes = cfg.MODEL.NUM_CLASSES

        # config for Context Refinement Decoder
        tubelet_size = cfg.ViT.TUBELET_SIZE
        patch_size = cfg.ViT.PATCH_SIZE
        pretrain_img_size = cfg.ViT.PRETRAIN_IMG_SIZE
        num_frames = cfg.DATA.NUM_FRAMES
        encoder_embed_dim = cfg.ViT.EMBED_DIM
        decoder_embed_dim = cfg.MODEL.CRD.EMBED_DIM
        decoder_depth = cfg.MODEL.CRD.DEPTH
        decoder_num_heads = cfg.MODEL.CRD.NUM_HEADS
        use_learnable_pos_emb = cfg.MODEL.CRD.USE_LEARNABLE_POS_EMB
        drop_rate = cfg.MODEL.CRD.DROP_RATE
        attn_drop_rate = cfg.MODEL.CRD.ATTN_DROP_RATE
        drop_path_rate = cfg.MODEL.CRD.DROP_PATH_RATE
        mlp_ratio = cfg.MODEL.CRD.MLP_RATIO
        use_checkpoint = cfg.ViT.USE_CHECKPOINT

        num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size) * (
                    num_frames // tubelet_size)
        self.grid_size = [pretrain_img_size // patch_size, pretrain_img_size // patch_size]

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim))
            trunc_normal_(self.pos_embed, std=.02)
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(num_patches, decoder_embed_dim)

        assert sum(dim_in) == encoder_embed_dim
        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        trunc_normal_(self.mask_token, std=.02)
        self.extension_scale = cfg.MODEL.CRD.ROI_EXTENSION

        self.decoder = ContextRefinementDecoder(
            cfg,
            patch_size=patch_size,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            use_checkpoint=use_checkpoint,
        )

        if self.use_focal:
            prior_prob = cfg.MODEL.SparseRCNN.PRIOR_PROB
            self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters()

    @staticmethod
    def _init_box_pooler(cfg):
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple([1.0 / x for x in [4, 8, 16, 32]])
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return box_pooler

    def _reset_parameters(self):
        # init all parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def bbox_extension(self, bboxes, images_whwh):
        scale = self.extension_scale
        if len(scale) < 2:
            x_scale = y_scale = scale[0]
        else:
            x_scale = scale[0]
            y_scale = scale[1]
        TO_REMOVE = 1
        xmin, ymin, xmax, ymax = bboxes.split(1, dim=-1)
        boxw, boxh = xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE
        padw, padh = float(x_scale) * boxw / 2, float(y_scale) * boxh / 2
        extended_xmin = xmin - padw
        extended_ymin = ymin - padh
        extended_xmax = xmax + padw
        extended_ymax = ymax + padh
        extended_box = torch.cat(
            (extended_xmin, extended_ymin, extended_xmax, extended_ymax), dim=-1
        )
        extended_box = clip_boxes_tensor(extended_box, images_whwh[1], images_whwh[0])
        return extended_box

    def forward(self, images_whwh, fpn_features, inputs, init_bboxes, init_features,
                ws, mask, criterion=None, targets=None):
        assert (
                len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)

        inter_class_logits = []
        inter_action_logits = []
        inter_pred_bboxes = []

        # reduce
        feat_pre_reduce = [feat for feat in inputs]

        if self.use_fpn:
            keyframe = fpn_features
        else:
            assert fpn_features is None, "Check Logic"

        x_vis = feat_pre_reduce[0]  # (B, N_vis, C_e)
        x_vis = self.encoder_to_decoder(x_vis)  # (B, N_vis, C_d)

        B, _, C = x_vis.shape
        num_s_tokens = ws[1] * ws[2]
        num_tokens = num_s_tokens * ws[0]

        pos_embed = self.pos_embed
        if self.pos_embed.shape[1] != num_tokens:
            pos_embed = interpolate_pos_embed_online(
                pos_embed, self.grid_size, [ws[1], ws[2]], 0).reshape(1, -1, C)

        key_start_idx = num_s_tokens * ws[0] // 2
        key_end_idx = num_s_tokens * (ws[0] // 2 + 1)
        expand_pos_embed = pos_embed.expand(B, -1, -1).type_as(x_vis).to(x_vis.device)

        # token remapping
        x_others = torch.zeros((B, num_tokens - num_s_tokens, C), dtype=x_vis.dtype, device=x_vis.device)
        x_key = x_vis[:, :num_s_tokens]
        x_nonkey = x_vis[:, num_s_tokens:]
        x_others[~mask] = x_nonkey.reshape(-1, C)
        x_others[mask] = self.mask_token.reshape(-1, C)
        x_full = torch.cat([x_others[:, :key_start_idx], x_key, x_others[:, key_start_idx:]], dim=1)

        x_full = x_full + expand_pos_embed
        pool_in = [x_full]

        # prepare decoder input
        key_pos_embed = expand_pos_embed[:, key_start_idx:key_end_idx]
        nonkey_pos_embed = torch.cat([expand_pos_embed[:, :key_start_idx], expand_pos_embed[:, key_end_idx:]], dim=1)
        vis_pos_emd = nonkey_pos_embed[~mask].reshape(B, -1, C)
        vis_pos_emd = torch.cat([key_pos_embed, vis_pos_emd], dim=1)
        x_vis = x_vis + vis_pos_emd  # (B, N_vis, C_d)

        # localization branch
        bboxes = init_bboxes
        # (num_proposals, 256) -> (1, num_proposals * B, 256)
        init_features = init_features[None].repeat(1, B, 1)
        proposal_features = init_features.clone()

        # code below this line, we assume pathway is 0.
        pathway = 0
        key_roi_align = getattr(self, "s{}_keyroi".format(pathway)) \
            if self.use_fpn else getattr(self, "s{}_roi".format(pathway))  # noqa
        for idx, rcnn_head in enumerate(getattr(self, "s{}_headseries".format(pathway))):
            class_logits, pred_bboxes, proposal_features, jitter_pred_bboxes = rcnn_head(keyframe, bboxes,
                                                                                         proposal_features,
                                                                                         key_roi_align,
                                                                                         images_whwh=images_whwh)
            if self.return_intermediate:
                inter_class_logits.append(class_logits)
                inter_pred_bboxes.append(pred_bboxes)
            bboxes = pred_bboxes.detach()

        if self.cfg.MODEL.SparseRCNN.JITTER_BOX:
            ava_box = jitter_pred_bboxes.detach()
        else:
            ava_box = bboxes
        if self.training:
            # fork person detector loss, matching indices, idx
            losses, indices, idx = self.person_detector_loss(inter_class_logits, inter_pred_bboxes, criterion, targets)

            # Use GT boxes to replace the corresponding position predicted box, with probability self.gt_boxes_prob
            if random.random() < self.gt_boxes_prob:  # random.random()  uniform ( 0 inclusive, 1 exclusive)
                ava_box = ava_box.clone()
                ava_box[idx] = torch.cat([t['boxes_xyxy'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # classification branch
        num_proposals = bboxes.shape[1]

        # roi extension
        bboxes = [self.bbox_extension(b, images_whwh=wh) for b, wh in zip(bboxes, images_whwh)]
        proposal_boxes = [Boxes(b) for b in bboxes]

        # roi align and pooling
        pool_out = []
        for i, po in enumerate(pool_in):
            po = po.reshape(B, ws[0], ws[1], ws[2], C).permute(0, 4, 1, 2, 3)  # (B, C_d, t, h, w)
            roi_align = getattr(self, "s{}_roi".format(i))

            num_frames = po.shape[2]
            rois = []
            for frame in range(num_frames):
                roi_per_frame = roi_align([po[:, :, frame]], proposal_boxes)  # (B*num_proposals, C_d, 7, 7)
                rois.append(roi_per_frame)
            out = torch.stack(rois, dim=2)  # (B*num_proposals, C_d, t, 7, 7)
            s_pool = getattr(self, "s{}_spool".format(i))
            out = s_pool(out)  # (B*num_proposals, C_d, t, 1, 1)

            t_pool = getattr(self, "s{}_tpool".format(i))
            out = t_pool(out)
            assert out.shape[2] == 1
            out = torch.squeeze(out, 2)  # (B*num_proposals, C_d, 1, 1)

            pool_out.append(out)
        roi_features = torch.cat(pool_out, 1)
        roi_features = roi_features.view(B, num_proposals, -1)  # (B, num_proposals, C_d)

        # conduct context refinement decoder
        dec_in = torch.cat([x_vis, roi_features], dim=1)  # (B, N_vis + num_proposals, C_d)
        action_logits = self.decoder(dec_in)  # (B, num_proposals, 80)

        if self.return_intermediate:
            inter_action_logits.extend(action_logits)

        if self.training:
            act_loss = self.action_cls_loss(inter_action_logits, targets, indices, idx)
            losses.update(act_loss)
            return losses

        return dict(pred_logits=inter_class_logits[-1],
                    pred_boxes=inter_pred_bboxes[-1],
                    pred_actions=inter_action_logits[-1])

    def person_detector_loss(self, outputs_class, outputs_coord, criterion, targets):
        if self.return_intermediate:
            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],
                      'aux_outputs': [{'pred_logits': a, 'pred_boxes': b}
                                      for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]}
        else:
            raise NotImplementedError

        loss_dict, indices, idx = criterion(output, targets)
        return loss_dict, indices, idx

    def action_cls_loss(self, output_action, targets, indices, idx):
        losses = {}
        target_actions_o = torch.cat([t["actions"][J] for t, (_, J) in zip(targets, indices)])
        for i, action_logits in enumerate(output_action):
            action = action_logits[idx]
            if not self.cfg.MODEL.SparseRCNN.SOFTMAX_POSE:
                if self.cfg.MODEL.LOSS_FUNC == 'focal_action':
                    act_loss = sigmoid_focal_loss_jit(action, target_actions_o, alpha=0.25, reduction='mean')
                else:
                    act_loss = F.binary_cross_entropy_with_logits(action, target_actions_o)  # remove Sigmoid in model
            else:
                pose_pred = F.softmax(action[:, :14], dim=-1)  # first 14 is pose label
                other_pred = F.sigmoid(action[:, 14:])
                action = torch.cat([pose_pred, other_pred], dim=-1)
                if self.cfg.MODEL.LOSS_FUNC == 'focal_action':
                    act_loss = focal_loss_wo_logits_jit(action, target_actions_o, alpha=0.25, reduction='mean')
                else:
                    act_loss = F.binary_cross_entropy(action, target_actions_o)

            losses.update({'loss_bce' + f'_{i}': act_loss})
        losses['loss_bce'] = losses.pop('loss_bce' + f'_{i}')  # modify the last loss key, making AVA meter happy
        return losses


class RCNNHead(nn.Module):

    def __init__(self, cfg, scale_clamp: float = _DEFAULT_SCALE_CLAMP, bbox_weights=(2.0, 2.0, 1.0, 1.0)):
        super().__init__()

        d_model = cfg.MODEL.SparseRCNN.HIDDEN_DIM
        num_classes = cfg.MODEL.SparseRCNN.NUM_CLASSES
        dim_feedforward = cfg.MODEL.SparseRCNN.DIM_FEEDFORWARD
        nhead = cfg.MODEL.SparseRCNN.NHEADS
        dropout = cfg.MODEL.SparseRCNN.DROPOUT
        activation = cfg.MODEL.SparseRCNN.ACTIVATION

        self.d_model = d_model
        self.jitter_box = cfg.MODEL.SparseRCNN.JITTER_BOX

        # dynamic.
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.inst_interact = DynamicConv(cfg)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        # cls.
        num_cls = cfg.MODEL.SparseRCNN.NUM_CLS
        cls_module = list()
        for _ in range(num_cls):
            cls_module.append(nn.Linear(d_model, d_model, False))
            cls_module.append(nn.LayerNorm(d_model))
            cls_module.append(nn.ReLU(inplace=True))
        self.cls_module = nn.ModuleList(cls_module)

        # reg.
        num_reg = cfg.MODEL.SparseRCNN.NUM_REG
        reg_module = list()
        for _ in range(num_reg):
            reg_module.append(nn.Linear(d_model, d_model, False))
            reg_module.append(nn.LayerNorm(d_model))
            reg_module.append(nn.ReLU(inplace=True))
        self.reg_module = nn.ModuleList(reg_module)

        # pred.
        self.use_focal = cfg.MODEL.SparseRCNN.USE_FOCAL
        if self.use_focal:
            self.class_logits = nn.Linear(d_model, num_classes)
            raise NotImplementedError
        else:
            assert num_classes == 1, "Check Person Detector num_classes {}".format(num_classes)
            self.class_logits = nn.Linear(d_model, num_classes + 1)

        self.bboxes_delta = nn.Linear(d_model, 4)
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights

    def forward(self, features, bboxes, pro_features, pooler, images_whwh=None):
        """
        :param bboxes: (N, nr_boxes, 4)
        :param pro_features: (N, nr_boxes, d_model)
        """

        N, nr_boxes = bboxes.shape[:2]

        # roi_feature.
        proposal_boxes = list()
        for b in range(N):
            proposal_boxes.append(Boxes(bboxes[b]))
        roi_features = pooler(features, proposal_boxes)
        roi_features = roi_features.view(N * nr_boxes, self.d_model, -1).permute(2, 0, 1)

        # self_att.
        pro_features = pro_features.view(N, nr_boxes, self.d_model).permute(1, 0, 2)
        pro_features2 = self.self_attn(pro_features, pro_features, value=pro_features)[0]
        pro_features = pro_features + self.dropout1(pro_features2)
        pro_features = self.norm1(pro_features)

        # inst_interact.
        pro_features = pro_features.view(nr_boxes, N, self.d_model).permute(1, 0, 2).reshape(1, N * nr_boxes,
                                                                                             self.d_model)
        pro_features2 = self.inst_interact(pro_features, roi_features)
        pro_features = pro_features + self.dropout2(pro_features2)
        obj_features = self.norm2(pro_features)

        # obj_feature.
        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)

        fc_feature = obj_features.transpose(0, 1).reshape(N * nr_boxes, -1)
        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)
        class_logits = self.class_logits(cls_feature)
        bboxes_deltas = self.bboxes_delta(reg_feature)
        pred_bboxes, jitter_pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4),
                                                            with_jitter=self.jitter_box, images_whwh=images_whwh,
                                                            N=N, nr_boxes=nr_boxes)

        return class_logits.view(N, nr_boxes, -1), pred_bboxes.view(N, nr_boxes, -1), obj_features, jitter_pred_bboxes

    def apply_deltas(self, deltas, boxes, with_jitter=False, images_whwh=None, N=None, nr_boxes=None):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.bbox_weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

        if not with_jitter:
            return pred_boxes, None

        jitter_pred_box = torch.zeros_like(deltas)
        if not self.training:
            # https://github.com/MVIG-SJTU/AlphAction/blob/master/alphaction/structures/bounding_box.py#L197
            x_scale = 0.1
            y_scale = 0.05
            jitter_pred_box[:, 0::4] = pred_ctr_x - 0.5 * pred_w * (1 + x_scale)
            jitter_pred_box[:, 1::4] = pred_ctr_y - 0.5 * pred_h * (1 + y_scale)
            jitter_pred_box[:, 2::4] = pred_ctr_x + 0.5 * pred_w * (1 + x_scale)
            jitter_pred_box[:, 3::4] = pred_ctr_y + 0.5 * pred_h * (1 + y_scale)
            jitter_pred_box = jitter_pred_box.view(N, nr_boxes, -1)
            for idx, (boxes_per_image, curr_whwh) in enumerate(zip(jitter_pred_box, images_whwh)):
                jitter_pred_box[idx] = clip_boxes_tensor(boxes_per_image, curr_whwh[1], curr_whwh[0])
            return pred_boxes, jitter_pred_box
        else:
            # https://github.com/MVIG-SJTU/AlphAction/blob/master/alphaction/structures/bounding_box.py#L226
            jitter_x_out, jitter_x_in, jitter_y_out, jitter_y_in = 0.2, 0.1, 0.1, 0.05
            device = pred_boxes.device

            def torch_uniform(rows, a=0.0, b=1.0):
                return torch.rand(rows, 1, dtype=torch.float32, device=device) * (b - a) + a

            num_boxes = N * nr_boxes

            jitter_pred_box[:, 0::4] = pred_ctr_x - 0.5 * pred_w + pred_w * torch_uniform(num_boxes, -jitter_x_out,
                                                                                          jitter_x_in)
            jitter_pred_box[:, 1::4] = pred_ctr_y - 0.5 * pred_h + pred_h * torch_uniform(num_boxes, -jitter_y_out,
                                                                                          jitter_y_in)
            jitter_pred_box[:, 2::4] = pred_ctr_x + 0.5 * pred_w + pred_w * torch_uniform(num_boxes, -jitter_x_in,
                                                                                          jitter_x_out)
            jitter_pred_box[:, 3::4] = pred_ctr_y + 0.5 * pred_h + pred_h * torch_uniform(num_boxes, -jitter_y_in,
                                                                                          jitter_y_out)
            jitter_pred_box = jitter_pred_box.view(N, nr_boxes, -1)
            for idx, (_, curr_whwh) in enumerate(zip(jitter_pred_box, images_whwh)):
                jitter_pred_box[idx][0].clamp_(min=0, max=curr_whwh[0] - 1)
                jitter_pred_box[idx][1].clamp_(min=0, max=curr_whwh[1] - 1)
                jitter_pred_box[idx][2] = torch.max(torch.clamp(jitter_pred_box[idx][2], max=curr_whwh[0] - 1),
                                                    jitter_pred_box[idx][0] + 1)
                jitter_pred_box[idx][3] = torch.max(torch.clamp(jitter_pred_box[idx][3], max=curr_whwh[1] - 1),
                                                    jitter_pred_box[idx][1] + 1)
                jitter_pred_box[idx] = clip_boxes_tensor(jitter_pred_box[idx], curr_whwh[1], curr_whwh[0])

            return pred_boxes, jitter_pred_box


class DynamicConv(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.hidden_dim = cfg.MODEL.SparseRCNN.HIDDEN_DIM
        self.dim_dynamic = cfg.MODEL.SparseRCNN.DIM_DYNAMIC
        self.num_dynamic = cfg.MODEL.SparseRCNN.NUM_DYNAMIC
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params)

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ReLU(inplace=True)

        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        num_output = self.hidden_dim * pooler_resolution ** 2
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features, roi_features):
        """
        pro_features: (1,  N * nr_boxes, self.d_model)
        roi_features: (49, N * nr_boxes, self.d_model)
        """
        features = roi_features.permute(1, 0, 2)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)

        param1 = parameters[:, :, :self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic, self.hidden_dim)

        features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)

        features = features.flatten(start_dim=1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.activation(features)

        return features


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
