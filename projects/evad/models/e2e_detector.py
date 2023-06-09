import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
from detectron2.structures import Boxes, Instances, ImageList
from detectron2.layers import Conv2d
from detectron2.modeling.backbone.fpn import LastLevelMaxPool

import slowfast.utils.weight_init_helper as init_helper
from slowfast.datasets.cv2_transform import clip_boxes_tensor
from slowfast.datasets.cv2_transform import detector_postprocess
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .loss import SetCriterion, HungarianMatcher


class E2EDetector(nn.Module):
    def __init__(self, cfg, num_pathway=1):
        super(E2EDetector, self).__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = num_pathway
        self.use_fpn = cfg.MODEL.SparseRCNN.USE_FPN
        if self.use_fpn:
            MAJOR_CUDA_VERSION, MINOR_CUDA_VERSION = torch.version.cuda.split('.')
            if cfg.MODEL.SparseRCNN.WORKAROUND or (MAJOR_CUDA_VERSION == '9' and MINOR_CUDA_VERSION == '0'):
                # workaround for https://github.com/pytorch/pytorch/issues/51333
                torch.backends.cudnn.deterministic = True
        self.cfg = cfg
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )
        if self.use_fpn:
            self._construct_fpn(cfg)
        self._construct_sparse(cfg)

    def _construct_sparse(self, cfg):
        self.num_proposals = cfg.MODEL.SparseRCNN.NUM_PROPOSALS
        self.hidden_dim = cfg.MODEL.SparseRCNN.HIDDEN_DIM
        self.num_heads = cfg.MODEL.SparseRCNN.NUM_HEADS
        self.decoder_depth = cfg.MODEL.CRD.DEPTH
        self.num_classes = cfg.MODEL.NUM_CLASSES

        # Build Proposals
        self.init_proposal_features = nn.Embedding(self.num_proposals, self.hidden_dim)
        self.init_proposal_boxes = nn.Embedding(self.num_proposals, 4)
        nn.init.constant_(self.init_proposal_boxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_boxes.weight[:, 2:], 1.0)

        # Loss Parameters:
        class_weight = cfg.MODEL.SparseRCNN.CLASS_WEIGHT
        giou_weight = cfg.MODEL.SparseRCNN.GIOU_WEIGHT
        l1_weight = cfg.MODEL.SparseRCNN.L1_WEIGHT
        action_weight = cfg.MODEL.SparseRCNN.ACTION_WEIGHT
        no_object_weight = cfg.MODEL.SparseRCNN.NO_OBJECT_WEIGHT
        self.deep_supervision = cfg.MODEL.SparseRCNN.DEEP_SUPERVISION
        self.use_focal = cfg.MODEL.SparseRCNN.USE_FOCAL

        # Build Criterion
        matcher = HungarianMatcher(cfg=cfg,
                                   cost_class=class_weight,
                                   cost_bbox=l1_weight,
                                   cost_giou=giou_weight,
                                   use_focal=self.use_focal)
        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight,
                       "loss_bce": action_weight}
        if self.deep_supervision:
            aux_weight_dict = {}
            for i in range(self.num_heads - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items() if k != 'loss_bce'})
            for i in range(self.decoder_depth - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items() if k == 'loss_bce'})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "boxes"]

        self.criterion = SetCriterion(cfg=cfg,
                                      num_classes=self.cfg.MODEL.SparseRCNN.NUM_CLASSES,  # person
                                      matcher=matcher,
                                      weight_dict=weight_dict,
                                      eos_coef=no_object_weight,
                                      losses=losses,
                                      use_focal=self.use_focal)
        self.to(self.device)

    def _construct_network(self, cfg):
        raise NotImplementedError

    def _construct_fpn(self, in_channels_per_feature):
        """ Default config follow SparseRCNN
        """
        out_channels = 256
        strides = [4, 8, 16, 32]  # different with original SlowFast that use DC5(Dilated-C5) for stage5
        lateral_convs = []
        output_convs = []

        norm = ""
        use_bias = norm == ""
        for idx, in_channels in enumerate(in_channels_per_feature):
            lateral_norm = None  # do not use norm as default now
            output_norm = None

            lateral_conv = Conv2d(
                in_channels, out_channels, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=use_bias, norm=output_norm
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            stage = int(math.log2(strides[idx]))
            self.add_module("fpn_lateral{}".format(stage), lateral_conv)
            self.add_module("fpn_output{}".format(stage), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        self.top_block = LastLevelMaxPool()
        in_features = ['res{}'.format(i) for i in range(2, 6)]
        self.in_features = in_features
        self._out_features = ['p{}'.format(i) for i in range(2, 7)]
        self._fuse_type = 'sum'
        self.rev_in_features = tuple(in_features[::-1])

    def fpn_forward(self, bottom_up_features):
        """copy Detectron2 FPN forward()
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.
                i.e. {'res2': Tensor, 'res3': Tensor, 'res4': Tensor, 'res5': Tensor}
        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        results = []
        prev_features = self.lateral_convs[0](bottom_up_features[self.in_features[-1]])
        results.append(self.output_convs[0](prev_features))

        # Reverse feature maps into top-down order (from low to high resolution)
        for features, lateral_conv, output_conv in zip(
                self.rev_in_features[1:], self.lateral_convs[1:], self.output_convs[1:]
        ):
            features = bottom_up_features[features]
            top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
            # Has to use explicit forward due to https://github.com/pytorch/pytorch/issues/47336
            lateral_features = lateral_conv.forward(features)
            prev_features = lateral_features + top_down_features
            if self._fuse_type == "avg":
                prev_features /= 2
            results.insert(0, output_conv.forward(prev_features))

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return dict(list(zip(self._out_features, results)))

    def forward(self, x, gt_instances):
        raise NotImplementedError

    def head_forward(self, x, images_whwh, gt_instances, fpn_output, window_size, mask):
        if self.use_fpn:
            # select 'p2', 'p3', 'p4', 'p5' following Sparse RCNN
            fpn_features = []
            for f in range(2, 6):
                fpn_features.append(fpn_output['p{}'.format(f)])
        else:
            fpn_features = None

        # Prepare Proposals
        proposal_boxes = self.init_proposal_boxes.weight.clone()
        proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes)
        proposal_boxes = proposal_boxes[None] * images_whwh[:, None, :]

        if self.training:
            targets = self.prepare_targets(gt_instances, images_whwh)
            loss_dict = self.det_head(images_whwh, fpn_features, x,
                                      proposal_boxes, self.init_proposal_features.weight,
                                      window_size, mask,
                                      self.criterion, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict

        else:
            output = self.det_head(images_whwh, fpn_features, x,
                                   proposal_boxes, self.init_proposal_features.weight,
                                   window_size, mask)
            person_logits = output["pred_logits"]  # (B, num_proposal, 2)
            boxes = output["pred_boxes"]  # (B, num_proposal, 4) Not normalized box (x1, y1, x2, y2)
            action_logits = output["pred_actions"]  # (B, num_proposal, 80)
            if self.cfg.MODEL.SparseRCNN.SOFTMAX_POSE:
                pose_logits = F.softmax(action_logits[:, :, :14], dim=-1)
                other_logits = torch.sigmoid(action_logits[:, :, 14:])
                action_logits = torch.cat([pose_logits, other_logits], dim=2)
            else:
                action_logits = torch.sigmoid(action_logits)

            if self.use_focal:
                raise NotImplementedError
            else:
                # For each box we assign the best class or the second best if the best on is `no_object`.
                scores, labels = F.softmax(person_logits, dim=-1)[:, :, :-1].max(-1)
                assert labels.sum() == 0, labels  # only "person" one class, id is 0

                all_action_boxes = []
                all_person_boxes = []
                all_actions = []
                all_metadata = []
                for i, (scores_per_image, boxes_per_image, actions_per_image, meta_per_image, curr_whwh) in enumerate(
                        zip(
                            scores, boxes, action_logits, gt_instances, images_whwh
                        )):

                    ori_h = meta_per_image["ori_height"]
                    ori_w = meta_per_image["ori_width"]

                    # for trivial_batch_collator, meta_per_image["metadata"] is e.g. [[0, 904]]
                    mtd = torch.tensor(meta_per_image["metadata"][0])
                    boxes_per_image = clip_boxes_tensor(boxes_per_image, curr_whwh[1], curr_whwh[0])
                    select_mask = scores_per_image >= self.cfg.MODEL.SparseRCNN.PERSON_THRESHOLD
                    if not any(select_mask):
                        """ if no, choose all """
                        select_mask = ~select_mask

                    select_actions = actions_per_image[select_mask]
                    if self.cfg.MODEL.SparseRCNN.NUM_EVAL_ACT_CLASSES > 0:
                        top_v, top_idx = torch.topk(select_actions, k=self.cfg.MODEL.SparseRCNN.NUM_EVAL_ACT_CLASSES,
                                                    dim=-1)
                        new_actions = torch.zeros_like(select_actions, dtype=select_actions.dtype,
                                                       device=select_actions.device)
                        new_actions.scatter_(dim=-1, index=top_idx, src=top_v)
                        select_actions = new_actions

                    select_boxes = boxes_per_image[select_mask]  # (num_select, 4)

                    # meta["metadata"] is used to tell gt current clip video idx, sec idx
                    all_metadata.append(mtd[None, :].repeat(len(select_boxes), 1))

                    select_boxes = detector_postprocess(select_boxes, ori_h, ori_w,
                                                        curr_whwh[1], curr_whwh[0],
                                                        norm_scale=True)
                    person_boxes = detector_postprocess(boxes_per_image, ori_h, ori_w,
                                                        curr_whwh[1], curr_whwh[0],
                                                        norm_scale=False)

                    # Put the image_id within a batch to the select_boxes[:, 0], which is not used for mAP, in fact.
                    select_boxes = torch.cat(
                        [
                            torch.full((select_boxes.shape[0], 1), float(i), device=select_mask.device),
                            select_boxes
                        ], dim=1
                    )  # (num_select, 5)

                    all_actions.append(select_actions)
                    all_action_boxes.append(select_boxes)
                    all_person_boxes.append(person_boxes)

                person_detector = dict(
                    boxes=torch.stack(all_person_boxes, dim=0),  # use stack keep the batch dim
                    scores=scores,
                )
                return torch.cat(all_actions, dim=0), torch.cat(all_action_boxes, dim=0), \
                       torch.cat(all_metadata, dim=0), person_detector

    def prepare_targets(self, targets, images_whwh):
        new_targets = []
        metas, labels = targets
        for meta_per_image, label_per_image, image_size_xyxy in zip(metas, labels, images_whwh):
            target = {}
            gt_classes = torch.tensor(label_per_image, dtype=torch.float32).to(self.device)
            meta_box = torch.tensor(meta_per_image['boxes'], dtype=torch.float32).to(self.device)
            gt_boxes = meta_box / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)

            # only one class  person: 0
            target["labels"] = torch.zeros(len(gt_boxes), dtype=torch.int64, device=self.device)
            target["actions"] = gt_classes.to(self.device)
            target["boxes"] = gt_boxes.to(self.device)
            target["boxes_xyxy"] = meta_box
            target["image_size_xyxy"] = image_size_xyxy.to(self.device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            new_targets.append(target)

        return new_targets

    def preprocess_image(self, batched_inputs):
        assert len(batched_inputs[0]) == 1 or len(batched_inputs[0]) == 2
        fast_on = len(batched_inputs[0]) == 2

        slow = [x[0].to(self.device) for x in batched_inputs]
        slow_images = ImageList.from_tensors(slow, size_divisibility=32)
        images = [slow_images.tensor]  # slow fast decorator
        if fast_on:
            fast = [x[1].to(self.device) for x in batched_inputs]
            fast_images = ImageList.from_tensors(fast, size_divisibility=32)
            images.append(fast_images.tensor)

        images_whwh = list()
        for bi in batched_inputs:
            h, w = bi[0].shape[-2:]  # size before the padding
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images_whwh = torch.stack(images_whwh)

        return images, images_whwh
