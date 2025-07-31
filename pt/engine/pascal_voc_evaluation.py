# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import logging
import numpy as np
import os
import tempfile
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from functools import lru_cache
import torch
import json
from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
import time
from detectron2.evaluation import DatasetEvaluator


class PascalVOCDetectionEvaluator(DatasetEvaluator):
    def __init__(self, cfg, dataset_name):
        """
        Args:
            dataset_name (str): name of the dataset, e.g., "voc_2007_test"
        """
        self._config_file = cfg
        self._dataset_name = dataset_name
        meta = MetadataCatalog.get(dataset_name)

        # Too many tiny files, download all to local for speed.
        annotation_dir_local = PathManager.get_local_path(
            os.path.join(meta.dirname, "Annotations/")
        )
        self._anno_file_template = os.path.join(annotation_dir_local, "{}.xml")
        self._image_set_path = os.path.join(meta.dirname, "ImageSets", "Main", meta.split + ".txt")
        self._class_names = meta.thing_classes
        # assert meta.year in [2007, 2012], meta.year
        self._is_2007 = meta.year == 2007
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

    def reset(self):
        self._predictions = defaultdict(list)  # class name -> list of prediction strings

    def process(self, inputs, outputs):
        print(f"[DEBUG] Processing {len(inputs)} images batch...")
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            instances = output["instances"].to(self._cpu_device)
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.tolist()
            classes = instances.pred_classes.tolist()
            for box, score, cls in zip(boxes, scores, classes):
                xmin, ymin, xmax, ymax = box
                # The inverse of data loading logic in `datasets/pascal_voc.py`
                xmin += 1
                ymin += 1
                self._predictions[cls].append(
                    f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
                )
        # print(f"[DEBUG] Predictions so far (per class): {[len(self._predictions[k]) for k in self._predictions]}")


    def evaluate(self):
        print("[DEBUG] Starting evaluation...")
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP", "AP50", and "AP75".
        """
        all_predictions = comm.gather(self._predictions, dst=0)
        # print(f"[DEBUG] Gathered predictions from all ranks. Main process: {comm.is_main_process()}")
        if not comm.is_main_process():
            return
        predictions = defaultdict(list)
        for predictions_per_rank in all_predictions:
            for clsid, lines in predictions_per_rank.items():
                predictions[clsid].extend(lines)
        # print(f"[DEBUG] Total predictions per class: {[len(predictions[k]) for k in predictions]}")
        del all_predictions

        self._logger.info(
            "Evaluating {} using {} metric. "
            "Note that results do not use the official Matlab API.".format(
                self._dataset_name, 2007 if self._is_2007 else 2012
            )
        )

        #將8個類別紀錄存到/home/u1755025/FedMPEN_mycode/output/class_predict/底下
        save_dir = os.path.join(self._config_file.MODEL.STORE_TP_FP_FN_ROOT_PATH, "class_predict")
        if not os.path.exists(save_dir):
            print(f"[DEBUG] Creating save_dir: {save_dir}")
            os.makedirs(save_dir)

        res_file_template = os.path.join(save_dir, "{}.txt") 
        aps = defaultdict(list)  # iou -> ap per class


        #定義並初始化image_tp_fp_fn_count存取tp、fp數量
        with PathManager.open(self._image_set_path, "r") as f:
            lines = f.readlines()
        imagenames = [x.strip() for x in lines]
        # print(f"[DEBUG] Loaded {len(imagenames)} image names from {self._image_set_path}")
        image_tp_fp_fn_count = {img: {"tp": 0, "fp": 0, "fn": 0} for img in imagenames}
        image_tp_fp_fn_bb = {img: {"tp": [], "fp": [], "fn": []} for img in imagenames}

        #創建兩個set儲存不重複的tp、fp座標
        store_tp_repeat = set()
        store_fp_repeat = set()
        store_fn_repeat = set()
        # print("[DEBUG] Initialized TP/FP/FN containers.")


        
        print(f"[DEBUG] Begin per-class evaluation for {len(self._class_names)} classes.")
        for cls_id, cls_name in enumerate(self._class_names):
            lines = predictions.get(cls_id, [""])
            print(f"[INFO] Class '{cls_name}': {len(lines)} predictions, start writing...")
            t_write0 = time.time()
            batch_size = 10000
            with open(res_file_template.format(cls_name), "w") as f:
                for i in range(0, len(lines), batch_size):
                    batch = lines[i:i+batch_size]
                    f.write("\n".join(batch))
                    f.write("\n")
                    f.flush()
                    print(f"  [WRITE] {cls_name}: wrote {min(i+batch_size, len(lines))}/{len(lines)} lines...")
            t_write1 = time.time()
            print(f"[INFO] Class '{cls_name}': writing done, time: {t_write1-t_write0:.2f}s")

            t_eval0 = time.time()
            for thresh in range(50, 100, 5):
                print(f"  [EVAL] {cls_name} thresh={thresh/100.0:.2f} ...", end="", flush=True)
                t0 = time.time()
                rec, prec, ap = voc_eval(
                    res_file_template,
                    self._anno_file_template,
                    self._image_set_path,
                    self._config_file,
                    cls_name,
                    ovthresh=thresh / 100.0,
                    use_07_metric=self._is_2007,
                    image_tp_fp_fn_count = image_tp_fp_fn_count,
                    image_tp_fp_fn_bb = image_tp_fp_fn_bb,
                    store_tp_repeat = store_tp_repeat,
                    store_fp_repeat = store_fp_repeat,
                    store_fn_repeat = store_fn_repeat,                   
                    store_ovthresh = self._config_file.MODEL.EVAL_OVTHRESH / 100.0
                )
                t1 = time.time()
                print(f" [EVAL] done. (time: {t1-t0:.2f}s, AP: {ap*100:.2f})")
                aps[thresh].append(ap * 100)
            t_eval1 = time.time()
            print(f"[INFO] Class '{cls_name}': all thresholds evaluated, time: {t_eval1-t_eval0:.2f}s\n")


        #將所有image的tp、fp數量記錄存到self._config_file.MODEL.STORE_TP_FP_FN_ROOT_PATH底下
        if self._config_file.MODEL.SAVE_TP_FP_FN == True:
            save_count = self._config_file.MODEL.STORE_TP_FP_FN_ROOT_PATH
            if not os.path.exists(save_count):
                os.makedirs(save_count)
            store_path_name = self._config_file.MODEL.STORE_TP_FP_FN_FILE_NAME
            output_file = os.path.join(save_count, f"{store_path_name}_count.txt")
            with open(output_file, "w") as file:
                for key, value in image_tp_fp_fn_count.items():
                    file.write(f"{key}: tp = {value['tp']}, fp = {value['fp']}, fn = {value['fn']}\n")

        #將所有image的tp、fp座標記錄存到self._config_file.MODEL.STORE_TP_FP_FN_ROOT_PATH底下
        if self._config_file.MODEL.SAVE_TP_FP_FN == True:

            save_bb = self._config_file.MODEL.STORE_TP_FP_FN_ROOT_PATH
            if not os.path.exists(save_bb):
                os.makedirs(save_bb)
            store_path_name = self._config_file.MODEL.STORE_TP_FP_FN_FILE_NAME  
            output_file_txt = os.path.join(save_bb, f"{store_path_name}_bb.txt")
            output_file_json = os.path.join(save_bb, f"{store_path_name}_bb.json")
            # 寫入 txt 格式（原本格式）
            with open(output_file_txt, "w") as file:
                for key, value in image_tp_fp_fn_bb.items():
                    tp_str = ', '.join([json.dumps(x, ensure_ascii=False) for x in value['tp']])
                    fp_str = ', '.join([json.dumps(x, ensure_ascii=False) for x in value['fp']])
                    fn_str = ', '.join([json.dumps(x, ensure_ascii=False) for x in value['fn']])
                    file.write(f"{key}: tp = [{tp_str}], fp = [{fp_str}], fn = [{fn_str}]\n")
            # 寫入 json 格式
            with open(output_file_json, "w") as fjson:
                json.dump(image_tp_fp_fn_bb, fjson, ensure_ascii=False, indent=2)


        ret = OrderedDict()
        mAP = {iou: np.mean(x) for iou, x in aps.items()}
        # print(f"[DEBUG] mAP summary: {ret}")
        ret["bbox"] = {"AP": np.mean(list(mAP.values())), "AP50": mAP[50], "AP75": mAP[75]}
        # print(f"[DEBUG] Evaluation complete. Returning results.")
        return ret

# #############################################################################
#
# Below code is modified from
# https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

"""Python implementation of the PASCAL VOC devkit's AP evaluation code."""


@lru_cache(maxsize=None)
def parse_rec(filename):
    # print(f"[DEBUG] Parsing annotation: {filename}")
    """Parse a PASCAL VOC xml file."""
    with PathManager.open(filename) as f:
        tree = ET.parse(f)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text if obj.find("name") is not None else "Unspecified"
        obj_struct["pose"] = obj.find("pose").text if obj.find("pose") is not None else "Unspecified"
        obj_struct["truncated"] = int(obj.find("truncated").text) if obj.find("truncated") is not None else 0
        obj_struct["difficult"] = int(obj.find("difficult").text) if obj.find("difficult") is not None else 0
        bbox = obj.find("bndbox")
        if bbox is not None:
            obj_struct["bbox"] = [
                int(bbox.find("xmin").text) if bbox.find("xmin") is not None else 0,
                int(bbox.find("ymin").text) if bbox.find("ymin") is not None else 0,
                int(bbox.find("xmax").text) if bbox.find("xmax") is not None else 0,
                int(bbox.find("ymax").text) if bbox.find("ymax") is not None else 0,
            ]
        else:
            obj_struct["bbox"] = [0, 0, 0, 0]
        objects.append(obj_struct)
    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath, annopath, imagesetfile, configfile, classname, image_tp_fp_fn_count, image_tp_fp_fn_bb, store_tp_repeat, store_fp_repeat, store_fn_repeat, store_ovthresh, ovthresh=0.5, use_07_metric=False):
    # print(f"[DEBUG] voc_eval: class={classname}, ovthresh={ovthresh}, detfile={detpath.format(classname)}")
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name

    # first load gt
    # read list of images
    with PathManager.open(imagesetfile, "r") as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    # print(f"[DEBUG] voc_eval: loaded {len(imagenames)} images from {imagesetfile}")

    # load annots
    recs = {}
    for idx, imagename in enumerate(imagenames):
        # if idx % 500 == 0:
            # print(f"[DEBUG] voc_eval: parsing annotation {idx+1}/{len(imagenames)}")
        recs[imagename] = parse_rec(annopath.format(imagename))

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] == classname]
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(bool)
        # difficult = np.array([False for x in R]).astype(bool)  # treat all "difficult" as GT
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}


    # read dets
    detfile = detpath.format(classname)
    with open(detfile, "r") as f:
        lines = f.readlines()
    # print(f"[DEBUG] voc_eval: loaded {len(lines)} detections from {detfile}")
    if len(lines) == 0:
        # print(f"[DEBUG] voc_eval: no detections for class {classname}")
        return np.array([]), np.array([]), 0.0

    splitlines = [x.strip().split(" ") for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 4)
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 4)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]


    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)



    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )


            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        #將bb轉成list形式
        bb_list = bb.tolist()
        bb_tuple = tuple(bb_list)

        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1
                    if ovthresh == store_ovthresh and configfile.MODEL.SAVE_TP_FP_FN == True:
                        if bb_tuple not in store_tp_repeat:
                            store_tp_repeat.add(bb_tuple)
                            image_tp_fp_fn_bb[image_ids[d]]["tp"].append({"bbox": bb_list, "pred_label": classname})
                            image_tp_fp_fn_count[image_ids[d]]["tp"] += 1
                else:
                    if ovthresh == store_ovthresh:
                        fp[d] = 1.0
                    if ovthresh == store_ovthresh and configfile.MODEL.SAVE_TP_FP_FN == True:
                        if bb_tuple not in store_fp_repeat:
                            store_fp_repeat.add(bb_tuple)
                            image_tp_fp_fn_bb[image_ids[d]]["fp"].append({"bbox": bb_list, "pred_label": classname})
                            image_tp_fp_fn_count[image_ids[d]]["fp"] += 1
        else:
            if ovthresh == store_ovthresh:
                fp[d] = 1.0
            if ovthresh == store_ovthresh and configfile.MODEL.SAVE_TP_FP_FN == True:
                if bb_tuple not in store_fp_repeat:
                    store_fp_repeat.add(bb_tuple)
                    image_tp_fp_fn_bb[image_ids[d]]["fp"].append({"bbox": bb_list, "pred_label": classname})
                    image_tp_fp_fn_count[image_ids[d]]["fp"] += 1

        # FN: 取正確 gt_label
        if ovthresh == store_ovthresh and configfile.MODEL.SAVE_TP_FP_FN == True:
            BBGT_list = BBGT.tolist()
            tp_bbox = []
            for i in range(len(R["det"])):
                if R["det"][i] == 1:
                    tp_bbox.append(BBGT_list[i])
                if not R["det"][i] and not R["difficult"][i]:
                    bbgt_tuple = tuple(BBGT_list[i])
                    if bbgt_tuple not in store_fn_repeat:
                        store_fn_repeat.add(bbgt_tuple)
                        # 取得正確 gt_label
                        gt_label = None
                        # recs[image_ids[d]] 是所有 GT 物件
                        for gt_obj in recs[image_ids[d]]:
                            if gt_obj["bbox"] == BBGT_list[i]:
                                gt_label = gt_obj["name"]
                                break
                        if gt_label is None:
                            gt_label = classname
                        image_tp_fp_fn_bb[image_ids[d]]["fn"].append({"bbox": BBGT_list[i], "gt_label": gt_label})
                        image_tp_fp_fn_count[image_ids[d]]["fn"] += 1
                # 刪除多出來的tp bbox(tp + fn = GT)，比對 bbox+label
                for TP in tp_bbox:
                    for idx, fn_item in enumerate(image_tp_fp_fn_bb[image_ids[d]]["fn"]):
                        if (
                            isinstance(fn_item, dict)
                            and fn_item.get("bbox") == TP
                            and fn_item.get("gt_label", classname) == classname
                        ):
                            image_tp_fp_fn_bb[image_ids[d]]["fn"].pop(idx)
                            image_tp_fp_fn_count[image_ids[d]]["fn"] -= 1
                            break




    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)


    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap
