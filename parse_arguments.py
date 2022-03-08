import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Getting the metrics and the figures - Precision(Recall), F1-score, ROC-curve.\n"
        "You can see the figures on your display or you can get them in .jpg format.\n"
        "You can get the metrics in terminal and like file in .csv format."
    )
    parser.add_argument(
        "--test_files",
        help="File with list of names of test samples in .txt format",
        type=str,
    )
    parser.add_argument(
        "--cls_names",
        help="file with the names of all classes in the format .json or .names",
        type=str,
    )
    parser.add_argument(
        "--xml_ann",
        help="absolute path to all files with annotations in the format .xml",
        type=str,
    )
    parser.add_argument(
        "--json_ann",
        help="absolute path to all files with annotations in the format .json",
        type=str,
    )
    parser.add_argument(
        "--coco_ann",
        help="absolute path to all files with annotations in the format .json\n"
             "(ATTENTION: native file format .json and COCO file format .json are different)",
        type=str,
    )
    parser.add_argument(
        "--ed_det",
        help="absolute path to all files with detections in the format .txt\n"
             "(ATTENTION: the number of files with detections must match the number of test images)",
        type=str,
    )
    parser.add_argument(
        "--yolo_det",
        help="absolute path to all files with detections in the format .txt\n"
             "(ATTENTION: the number of files with detections must match the number of classes,\n"
             " and the names of the classes must be indicated in the file names)",
        type=str,
    )
    parser.add_argument(
        "--show_fig",
        action='store_true',
        help="show graphs of Precision(Recall), F1-score, ROC-curve functions for each class",
    )
    parser.add_argument(
        "--save_fig",
        action='store_true',
        help="save graphs of Precision(Recall), F1-score, ROC-curve functions for each class in the format .jpg",
    )
    parser.add_argument(
        "--show_term",
        action='store_true',
        help="show the results of calculating F1, Recall, Precision, AUC, AP, mAP metrics in the terminal window",
    )
    parser.add_argument(
        "--save_metric",
        action='store_true',
        help="save the results of calculating F1, Recall, Precision, AUC, AP, mAP metrics to a file in the format .csv",
    )
    parser.add_argument(
        "--yolo",
        action='store_true',
        help="configuration of parsers for detector YOLO",
    )
    parser.add_argument(
        "--efficient_det",
        action='store_true',
        help="configuration of parsers for detector EfficientDet",
    )

    return parser.parse_args()