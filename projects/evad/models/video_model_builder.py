from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from slowfast.models.build import MODEL_REGISTRY
from .e2e_detector import E2EDetector
from .head_helper import EVADRoIHead
from .vit_model import PatchEmbed, KTPBlock, get_sinusoid_encoding_table, interpolate_pos_embed_online


@MODEL_REGISTRY.register()
class EVAD(E2EDetector):
    def __init__(self, cfg):
        nn.Module.__init__(self)  # call grandparent's __init__ w/o calling parent's.
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 1
        self.use_fpn = cfg.MODEL.SparseRCNN.USE_FPN
        if self.use_fpn:
            MAJOR_CUDA_VERSION, MINOR_CUDA_VERSION = torch.version.cuda.split('.')
            if cfg.MODEL.SparseRCNN.WORKAROUND or (MAJOR_CUDA_VERSION == '9' and MINOR_CUDA_VERSION == '0'):
                # workaround for https://github.com/pytorch/pytorch/issues/51333
                torch.backends.cudnn.deterministic = True
        embed_dim = cfg.ViT.EMBED_DIM
        self.num_features = [embed_dim]
        self.cfg = cfg
        self._construct_network(cfg)
        if self.use_fpn:
            self.fpn_indices = cfg.ViT.FPN_INDICES
            num_features = [embed_dim for _ in range(4)]
            self.num_features = num_features
            self._construct_fpn(in_channels_per_feature=num_features)
        self._construct_sparse(cfg)

    def _construct_network(self, cfg):
        """
        Builds a single pathway ViT model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        tubelet_size = cfg.ViT.TUBELET_SIZE
        patch_size = cfg.ViT.PATCH_SIZE
        embed_dim = cfg.ViT.EMBED_DIM
        pretrain_img_size = cfg.ViT.PRETRAIN_IMG_SIZE
        use_learnable_pos_emb = cfg.ViT.USE_LEARNABLE_POS_EMB
        drop_rate = cfg.ViT.DROP_RATE
        attn_drop_rate = cfg.ViT.ATTN_DROP_RATE
        drop_path_rate = cfg.ViT.DROP_PATH_RATE
        depth = cfg.ViT.DEPTH
        num_heads = cfg.ViT.NUM_HEADS
        mlp_ratio = cfg.ViT.MLP_RATIO
        qkv_bias = cfg.ViT.QKV_BIAS
        qk_scale = cfg.ViT.QK_SCALE
        init_values = cfg.ViT.INIT_VALUES
        use_checkpoint = cfg.ViT.USE_CHECKPOINT
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.patch_embed = PatchEmbed(
            img_size=pretrain_img_size, patch_size=patch_size, in_chans=cfg.DATA.INPUT_CHANNEL_NUM[0],
            embed_dim=embed_dim, tubelet_size=tubelet_size, num_frames=cfg.DATA.NUM_FRAMES)
        num_patches = self.patch_embed.num_patches
        self.grid_size = [pretrain_img_size // patch_size, pretrain_img_size // patch_size]
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.pos_embed, std=.02)
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        keep_rates = cfg.MODEL.KTP.KEEP_RATES
        assert len(keep_rates) == depth
        enhanced_weight = cfg.MODEL.KTP.ENHANCED_WEIGHT

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            KTPBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, use_checkpoint=use_checkpoint,
                keep_rate=keep_rates[i], enhanced_weight=enhanced_weight)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        if self.enable_detection:
            self.det_head = EVADRoIHead(cfg, dim_in=[self.num_features[-1]],)
        else:
            raise NotImplementedError

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x_list, gt_instances):

        x_list, images_whwh = self.preprocess_image(x_list)
        x = x_list[0]  # (B, 3, T, H, W)

        x = self.patch_embed(x)  # (B, C_e, t, h, w)
        ws = x.shape[2:]

        num_frames = x.shape[2]
        assert num_frames % 2 == 0, "Only consider even case, check frames {}".format(num_frames)
        key_idx = num_frames // 2

        x = x.flatten(2).transpose(1, 2)  # (B, N, C_e), N = t x h x w
        B, _, C = x.shape

        pos_embed = self.pos_embed
        if self.pos_embed.shape[1] != x.shape[1]:
            pos_embed = interpolate_pos_embed_online(
                pos_embed, self.grid_size, [ws[1], ws[2]], 0).reshape(1, -1, C)

        x = x + pos_embed.type_as(x).to(x.device)
        x = self.pos_drop(x)

        # move keyframe tokens to the first column
        x = x.reshape(B, num_frames, -1, C)
        x = torch.cat([x[:, key_idx].unsqueeze(1), x[:, :key_idx], x[:, key_idx+1:]], dim=1)
        x = x.reshape(B, -1, C)

        # keep the global indexes of non-keyframe tokens during pruning
        num_s_tokens = ws[1] * ws[2]
        num_tokens = num_s_tokens * ws[0]
        idx = torch.arange(0, num_tokens-num_s_tokens, device=x.device).unsqueeze(0).repeat(B, 1)

        fpn_features = dict()
        for i in range(self.depth):
            blk = self.blocks[i]
            x, idx = blk(x, idx, ws)

            if self.use_fpn:
                if i in self.fpn_indices:
                    x_key = x[:, :num_s_tokens].reshape(B, ws[1], ws[2], -1).permute(0, 3, 1, 2)
                    if i == self.fpn_indices[0]:
                        x_key = F.interpolate(x_key, scale_factor=4.0, mode="nearest")
                    elif i == self.fpn_indices[1]:
                        x_key = F.interpolate(x_key, scale_factor=2.0, mode="nearest")
                    elif i == self.fpn_indices[-1]:
                        x_key = F.interpolate(x_key, scale_factor=0.5, mode="nearest")
                    fpn_features.update({'res{}'.format(self.fpn_indices.index(i) + 2): x_key})
        fpn_output = self.fpn_forward(fpn_features)

        # generate masking based on the preserved indexes
        mask = torch.ones((B, num_tokens-num_s_tokens), dtype=torch.bool, device=x.device)
        bs_idx = torch.arange(B, device=x.device)
        mask[bs_idx, idx.transpose(0, 1)] = False

        x = self.norm(x)  # (B, N_vis, C_e)
        x = [x]
        return self.head_forward(x, images_whwh, gt_instances, fpn_output, window_size=ws, mask=mask)
