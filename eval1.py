
import os
import argparse
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.engine import default_setup, launch
import sys
# --- 引用自專案的必要模組 ---
from pt.engine.FLtrainer import FLtrainer
from pt.engine.trainer_sourceonly import PTrainer_sourceonly
from pt import add_config
from FLpkg import FedUtils, add_config as FL_add_config

# --- to register ---
from pt.modeling.meta_arch.rcnn import GuassianGeneralizedRCNN
from pt.modeling.proposal_generator.rpn import GuassianRPN
from pt.modeling.roi_heads.roi_heads import GuassianROIHead
import pt.data.datasets.builtin
from pt.modeling.backbone.vgg import build_vgg_backbone
from pt.modeling.anchor_generator import DifferentiableAnchorGenerator

def setup_cfg(args, model_weight_path, model_output_dir):
    """
    為單次評估建立並設定 config 物件。
    """
    cfg = get_cfg()
    add_config(cfg)
    FL_add_config(cfg)
    cfg.merge_from_file(args.config_file)
    
    # --- 最關鍵的步驟：為本次評估覆寫特定參數 ---
    opts = [
        'MODEL.WEIGHTS', model_weight_path,
        'OUTPUT_DIR', model_output_dir,
        'MODEL.STORE_TP_FP_FN_ROOT_PATH', model_output_dir
    ]
    if args.opts:
        opts.extend(args.opts)
        
    cfg.merge_from_list(opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def do_single_evaluation(cfg):
    """
    執行單一模型的評估核心邏輯。
    此函式整合並精簡了 eval.py 的 main 函式內容。
    """
    # 根據 config 決定 Trainer 類型
    if cfg.MODEL.STUDENT_TRAINER == "sourceonly":
        Trainer = PTrainer_sourceonly
    else:
        Trainer = FLtrainer

    # --- 精簡後的評估邏輯 (僅保留 FLtrainer 分支) ---
    model_final_output = torch.load(cfg.MODEL.WEIGHTS, map_location=torch.device("cpu"))
    
    if 'model' in model_final_output:
        state_dict = model_final_output['model']
    else:
        state_dict = model_final_output
        
    model_with_student_prefix = {
        k: v for k, v in state_dict.items() if k.startswith('modelStudent.')
    }
    model_wo_student_prefix = {
        k.replace('modelStudent.', ''): v for k, v in model_with_student_prefix.items()
    }
    model_wo_student_prefix_ordered = OrderedDict(model_wo_student_prefix)

    # 建立模型
    if cfg.FEDSET.DYNAMIC:
        backbone_dim = FedUtils.get_backbone_shape(model_wo_student_prefix)
        cfg.defrost()
        cfg.BACKBONE_DIM = backbone_dim
        cfg.freeze()
        model = Trainer.build_model(cfg, cfg.BACKBONE_DIM, False)
    else:
        model = Trainer.build_model(cfg)
    
    # 載入權重並執行測試
    model.load_state_dict(model_wo_student_prefix_ordered, strict=False)
    res = Trainer.test(cfg, model)
    return res

def main_loop(args):
    """
    主迴圈，遍歷所有模型權重並逐一進行評估。
    這個函式將被 `launch` 呼叫，以支援 multi-GPU。
    """
    # 檢查權重資料夾是否存在
    if not os.path.isdir(args.weights_dir):
        if comm.is_main_process():
            print(f"錯誤：權重資料夾 '{args.weights_dir}' 不存在。")
        return

# ... 其餘程式碼請依原始 eval1.py 補齊 ...
