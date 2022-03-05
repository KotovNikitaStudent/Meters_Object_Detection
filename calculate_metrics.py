import numpy as np
from compute_overlap import compute_overlap


def calculate_metrics(data):
    """Calculation of F1, R, P, score metrics, recording data about them in the .json file"""
    det = data["detections"]
    ann = data["annotations"]

    pr, rc, f1_m, sc, ap, tp_roc, fp_roc = [], [], [], [], [], [], []
    average_precisions = 0
    iou_threshold = 0.5

    all_detections = [list(i.values()) for i in det]
    all_annotations = [list(i.values()) for i in ann]

    for label in range(len(all_detections[0])):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0

        for i in range(len(all_detections)):

            if len(all_annotations[i][label]) != 0:
                annotations = all_annotations[i][label]
            else:
                continue

            if len(all_detections[i][label]) != 0:
                detections = all_detections[i][label]
            else:
                continue

            num_annotations += len(annotations)
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])
                d = np.asarray([d])
                annotations = np.array(
                    [np.array(xi) for xi in annotations], dtype=np.float64
                )

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(d, annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if (
                    max_overlap >= iou_threshold
                    and assigned_annotation not in detected_annotations
                ):
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        if num_annotations == 0:
            average_precisions = 0
            continue

        true_positive_roc = true_positives
        false_positives_roc = false_positives

        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(
            true_positives + false_positives, np.finfo(np.float64).eps
        )

        pr.append(precision.tolist())
        rc.append(recall.tolist())

        f1 = 2.0 * (precision * recall) / (precision + recall + 1e-9)
        f1_m.append(f1.tolist())
        sc.append(np.sort(scores).tolist())
        average_precisions = compute_ap(recall, precision)
        ap.append(average_precisions)

        tp_roc.append(true_positive_roc.tolist())
        fp_roc.append(false_positives_roc.tolist())

    result = {}
    result.update(
        {
            "pr": pr,
            "rc": rc,
            "sc": sc,
            "f1_m": f1_m,
            "ap": ap,
            "tp_roc": tp_roc,
            "fp_roc": fp_roc,
        }
    )
    return result


def compute_ap(recall: list, precision: list):
    """Compute the average precision, given the recall and precision curves"""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap.tolist()
