from fvcore.common.config import CfgNode as CN


def add_evad_config(cfg):
    # Add config for SparseRCNN.
    cfg.MODEL.SparseRCNN = CN()
    cfg.MODEL.SparseRCNN.NUM_CLASSES = 1
    cfg.MODEL.SparseRCNN.NUM_PROPOSALS = 100

    # RCNN Head.
    cfg.MODEL.SparseRCNN.NHEADS = 8
    cfg.MODEL.SparseRCNN.DROPOUT = 0.0
    cfg.MODEL.SparseRCNN.DIM_FEEDFORWARD = 2048
    cfg.MODEL.SparseRCNN.ACTIVATION = 'relu'
    cfg.MODEL.SparseRCNN.HIDDEN_DIM = 256
    cfg.MODEL.SparseRCNN.NUM_CLS = 1
    cfg.MODEL.SparseRCNN.NUM_ACT = 1
    cfg.MODEL.SparseRCNN.NUM_REG = 3
    cfg.MODEL.SparseRCNN.NUM_HEADS = 6

    # Dynamic Conv.
    cfg.MODEL.SparseRCNN.NUM_DYNAMIC = 2
    cfg.MODEL.SparseRCNN.DIM_DYNAMIC = 64

    # Loss.
    cfg.MODEL.SparseRCNN.CLASS_WEIGHT = 2.0
    cfg.MODEL.SparseRCNN.GIOU_WEIGHT = 2.0
    cfg.MODEL.SparseRCNN.L1_WEIGHT = 5.0
    cfg.MODEL.SparseRCNN.ACTION_WEIGHT = 12.0
    cfg.MODEL.SparseRCNN.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.SparseRCNN.DEEP_SUPERVISION = True

    # Focal Loss.
    cfg.MODEL.SparseRCNN.USE_FOCAL = False
    cfg.MODEL.SparseRCNN.ALPHA = 0.25
    cfg.MODEL.SparseRCNN.GAMMA = 2.0
    cfg.MODEL.SparseRCNN.PRIOR_PROB = 0.01

    # Custom Additional.
    cfg.MODEL.SparseRCNN.PERSON_THRESHOLD = 0.7
    cfg.MODEL.SparseRCNN.JITTER_BOX = False  # use refined box for action roi_feature
    cfg.MODEL.SparseRCNN.SOFTMAX_POSE = False
    cfg.MODEL.SparseRCNN.GT_BOXES_PROB = 0.
    cfg.MODEL.SparseRCNN.KEYWAY = False
    cfg.MODEL.SparseRCNN.USE_FPN = True
    cfg.MODEL.SparseRCNN.ACT_FC_DIM = 2048
    cfg.MODEL.SparseRCNN.PONY = False
    cfg.MODEL.SparseRCNN.WORKAROUND = False
    cfg.MODEL.SparseRCNN.NUM_EVAL_ACT_CLASSES = 0

    # Add config for Keyframe-centric Token Pruning (KTP).
    cfg.MODEL.KTP = CN()
    cfg.MODEL.KTP.KEEP_RATES = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    cfg.MODEL.KTP.ENHANCED_WEIGHT = 1

    # Add config for Context Refinement Decoder.
    cfg.MODEL.CRD = CN()
    cfg.MODEL.CRD.ROI_EXTENSION = (0., 0.)
    cfg.MODEL.CRD.EMBED_DIM = 384
    cfg.MODEL.CRD.DEPTH = 6
    cfg.MODEL.CRD.NUM_HEADS = 6
    cfg.MODEL.CRD.USE_LEARNABLE_POS_EMB = False
    cfg.MODEL.CRD.DROP_RATE = 0.
    cfg.MODEL.CRD.ATTN_DROP_RATE = 0.
    cfg.MODEL.CRD.DROP_PATH_RATE = 0.
    cfg.MODEL.CRD.MLP_RATIO = 4
