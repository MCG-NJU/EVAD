import os
import simplejson as json
from detectron2.utils import comm
from tqdm import tqdm
import slowfast.utils.logging as logging
import slowfast.utils.distributed as du


class PersonMeter(object):
    def __init__(self, overall_iter, cfg, mode='test'):

        self.cfg = cfg
        self.mode = mode
        self.all_preds = []
        if cfg.AVA.FULL_TEST_ON_VAL or mode == 'test':
            split = "person_{}2020.json".format('test')
        else:
            split = "person_{}2020.json".format(mode)
        if cfg.AVA.GROUNDTRUTH_FILE == "ava_val_v2.2.csv":
            person_version = 'anno_person'
        elif cfg.AVA.GROUNDTRUTH_FILE == "ava_val_v2.1.csv":
            person_version = 'anno_person2v1'
        else:
            raise ValueError(cfg.AVA.GROUNDTRUTH_FILE)
        self.gt_path = os.path.join(cfg.DATA.PATH_TO_DATA_DIR, person_version, split)
        self.mapping_path = os.path.join(cfg.DATA.PATH_TO_DATA_DIR, person_version, "ava_test_mapping.json")
        """
        mapping: List[dict]
        {'idx': 0, 'filename': '1j20qq1JyX4_000061.jpg', 'sec': 902, 'height': 480, 'width': 600}
        """
        self.reset()

    def reset(self):
        self.all_preds = []

    def convert_pred_format(self, pred):
        """
        input format:
        output format:
        """

        return

    def update_stats(self, boxes, scores, file_ids):
        """
        :param boxes: (Batch, num_box, 4)
        :param scores: (Batch, num_box)
        :param file_ids: (Batch)
        :return:
        """
        assert boxes.dim() == 3, "only support multi gpu test, current box shape {}".format(boxes.shape)
        assert len(boxes) == len(scores) == len(file_ids), \
            "{} vs {} vs {}".format(len(boxes), len(scores), len(file_ids))
        for boxes_per_image, scores_per_image, file_id in zip(boxes, scores, file_ids):
            assert len(boxes_per_image) == len(scores_per_image), \
            "{} vs {}".format(len(boxes_per_image), len(scores_per_image))
            for box, score in zip(boxes_per_image, scores_per_image):
                left, top, right, bottom = box
                # bbox = left + " " + top + " " + right + " " + bottom
                bbox = "{} {} {} {}".format(left, top, right, bottom)
                self.all_preds.append(dict(
                    confidence=score,
                    file_id=file_id,
                    bbox=bbox,
                ))

    def prepare_gt(self, json_path):
        print("loading ground truth person from ")
        """
         ground-truth
             Load each of the ground-truth files into a temporary ".json" file.
             Create a list of all the class names present in the ground-truth (gt_classes).
        """
        with open(json_path) as json_file:
            data = json.load(json_file)

        images = data['images']
        images_id_to_filename = []
        for idx, image in enumerate(images):
            assert idx == image["id"], "{} vs {}".format(idx, image["id"])
            images_id_to_filename.append(image["file_name"])

        categories = data['categories']
        class_name = categories[0]['name']
        assert class_name == 'person'
        obj_count = 0

        annotations = data['annotations']
        bounding_boxes = dict()
        for data_anno in annotations:
            # not norm, relate to original image (saved in disk)
            file_name = images_id_to_filename[data_anno["image_id"]]
            x_min, y_min, box_w, box_h = data_anno['bbox']
            x_max = x_min + box_w
            y_max = y_min + box_h
            # bbox = x_min + " " + y_min + " " + x_max + " " + y_max
            bbox = "{} {} {} {}".format(x_min, y_min, x_max, y_max)
            one_box = dict(
                class_name=class_name,
                bbox=bbox,
                used=False,
            )
            obj_count += 1
            bounding_boxes.setdefault(file_name, []).append(one_box)
        return bounding_boxes, obj_count

    def save_json(self, data, mapping_path, save_path=None):
        """
        save detection results to coco format json,
        {'image_id': 0,
        'category_id': 0,
        'bbox': [260.76, 267.45, 166.35 204.13], # [leftx1, topy1, width, height]
        'score': 0.0007702982984483242}

        call this func after clean the prediction data.
        """
        coco_instances = []
        with open(mapping_path) as fh:
            mapping = json.load(fh)

        filenames_to_idx = dict()
        for mp in mapping:
            filenames_to_idx[mp['filename']] = mp['idx']

        for d in tqdm(data):
            x1, y1, x2, y2 = [float(x) for x in d['bbox'].split()]
            w = x2 - x1
            h = y2 - y1
            coco_instances.append(dict(
                image_id=filenames_to_idx[d['file_id']],
                category_id=0,  # only person
                bbox=[x1, y1, w, h],
                score=float(d['confidence']),
            ))
        if save_path is None:
            # don't save intermediate results. Now, if save, it may assert error for multiprocess.
            # maybe related to https://github.com/AirtestProject/Airtest/issues/325#issuecomment-473744869
            return coco_instances

        with open(save_path, 'w') as jh:
            json.dump(coco_instances, jh)
        return save_path

    def finalize_metrics(self, log=True, MINOVERLAP=0.5, coco_api=True):
        comm.synchronize()
        if not comm.is_main_process():
            return
        class_name = "person"
        count_true_positives = 0
        # Load detection-results of that class
        dr_data = self.all_preds
        # sort detection-results by decreasing confidence
        dr_data.sort(key=lambda x:float(x['confidence']), reverse=True)

        # Load ground truth
        gt_data, gt_counter_person = self.prepare_gt(self.gt_path)
        gt_data_filenames = gt_data.keys()

        # some frames do not have gt, so we skip these prediction
        dr_data = [dr for dr in dr_data if dr['file_id'] in gt_data_filenames]

        # check if all ground truth images have been tested
        det_file_ids = set([d["file_id"] for d in dr_data])
        assert len(det_file_ids) == len(gt_data_filenames), "{} vs {}".format(len(det_file_ids), len(gt_data_filenames))

        if coco_api:
            coco_format_ava = self.save_json(dr_data, self.mapping_path)
            from pycocotools.coco import COCO
            from .cocoeval import COCOeval, create_small_table
            cocoGt = COCO(self.gt_path)
            cocoDt = cocoGt.loadRes(coco_format_ava)

            coco_eval = COCOeval(cocoGt, cocoDt, iouType='bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl"]
            results = dict()
            for m, v in zip(metrics, coco_eval.stats[:6]):
                results[m] = v * 100
            table_res = create_small_table(results)
            if log:
                log_msg = ""
                for k in ["AP", "AP50", "AP75"]:
                    log_msg += "{}:{}  ".format(k, results[k])
                stats = {"mode": self.mode, "Person Detector": log_msg}
                logging.log_json_stats(stats)
                if du.is_master_proc():
                    with open("{}/ap_summary.txt".format(self.cfg.OUTPUT_DIR), 'a') as fh:
                        fh.write("Person Detection COCO AP:\n")
                        fh.write("{}\n".format(table_res))
            return

        # Assign detection-results to ground-truth objects
        nd = len(dr_data)
        tp = [0] * nd # creates an array of zeros of size nd
        fp = [0] * nd
        for idx, detection in enumerate(tqdm(dr_data, desc="Person Detections")):
            file_id = detection["file_id"]
            # assign detection-results to ground truth object if any
            # open ground-truth with that file_id
            ground_truth_data = gt_data[file_id]
            ovmax = -1
            gt_match = -1
            # load detected object bounding-box
            bb = [ float(x) for x in detection["bbox"].split() ]
            for obj in ground_truth_data:
                # look for a class_name match
                assert obj["class_name"] == class_name, obj["class_name"]
                bbgt = [float(x) for x in obj["bbox"].split()]
                bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                iw = bi[2] - bi[0] + 1
                ih = bi[3] - bi[1] + 1
                if iw > 0 and ih > 0:
                    # compute overlap (IoU) = area of intersection / area of union
                    ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                    + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                    ov = iw * ih / ua
                    if ov > ovmax:
                        ovmax = ov
                        gt_match = obj

            # set minimum overlap
            min_overlap = MINOVERLAP
            if ovmax >= min_overlap:
                if "difficult" not in gt_match:
                    if not bool(gt_match["used"]):
                        # true positive
                        tp[idx] = 1
                        gt_match["used"] = True
                        count_true_positives += 1
                    else:
                        # false positive (multiple detection)
                        fp[idx] = 1
            else:
                # false positive
                fp[idx] = 1
                if ovmax > 0:
                    status = "INSUFFICIENT OVERLAP"

        # compute precision/recall
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val
        #print(tp)
        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / gt_counter_person
        #print(rec)
        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
        #print(prec)

        ap, mrec, mprec = self.voc_ap(rec[:], prec[:])
        if log:
            stats = {"mode": self.mode, "Person Detector@{}IOU".format(MINOVERLAP): ap}
            logging.log_json_stats(stats)
            if du.is_master_proc():
                with open("{}/ap_summary.txt".format(self.cfg.OUTPUT_DIR), 'a') as fh:
                    fh.write("Person AP@{}: {}\n".format(MINOVERLAP, ap*100))

    def voc_ap(self, rec, prec):
        """
        --- Official matlab code VOC2012---
        mrec=[0 ; rec ; 1];
        mpre=[0 ; prec ; 0];
        for i=numel(mpre)-1:-1:1
                mpre(i)=max(mpre(i),mpre(i+1));
        end
        i=find(mrec(2:end)~=mrec(1:end-1))+1;
        ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
        """
        rec.insert(0, 0.0) # insert 0.0 at begining of list
        rec.append(1.0) # insert 1.0 at end of list
        mrec = rec[:]
        prec.insert(0, 0.0) # insert 0.0 at begining of list
        prec.append(0.0) # insert 0.0 at end of list
        mpre = prec[:]
        """
         This part makes the precision monotonically decreasing
            (goes from the end to the beginning)
            matlab: for i=numel(mpre)-1:-1:1
                        mpre(i)=max(mpre(i),mpre(i+1));
        """
        # matlab indexes start in 1 but python in 0, so I have to do:
        #     range(start=(len(mpre) - 2), end=0, step=-1)
        # also the python function range excludes the end, resulting in:
        #     range(start=(len(mpre) - 2), end=-1, step=-1)
        for i in range(len(mpre)-2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i+1])
        """
         This part creates a list of indexes where the recall changes
            matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
        """
        i_list = []
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i-1]:
                i_list.append(i) # if it was matlab would be i + 1
        """
         The Average Precision (AP) is the area under the curve
            (numerical integration)
            matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
        """
        ap = 0.0
        for i in i_list:
            ap += ((mrec[i]-mrec[i-1])*mpre[i])
        return ap, mrec, mpre

