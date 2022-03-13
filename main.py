import os

from Preprocessing.Extract_Annotation import *
from Preprocessing.Extract_Detection import *
from Preprocessing.PreprocessData import EfficientDetPreprocessor, YOLOPreprocessor
from utils.calculate_metrics import calculate_metrics
from utils.figure import figure
from utils.Writer import Writer
from utils.parse_arguments import parse_arguments
from utils.extract_class_names import extract_class_names_from_file


def main() -> None:
    """
    Main function
    """
    
    preprocess_dict = {
        "classes": None,
        "ann_extractor": None,
        "det_extractor": None,
        "ann_path": None,
        "det_path": None,
        "names_test_samples": None,
    }
    preprocess = None

    args = parse_arguments()

    if args.test_files is not None:
        preprocess_dict["names_test_samples"] = args.test_files
    else:
        raise Exception(
            "You did not specify the absolute path to the file with all test instances"
        )

    if args.cls_names is not None:
        preprocess_dict["classes"] = extract_class_names_from_file(args.cls_names)
    else:
        raise Exception(
            "You did not specify an absolute path to the file with class names"
        )

    if args.yolo_det is not None:
        if not os.path.exists(args.yolo_det):
            raise Exception("The specified directory does not exist before files with detections")
        preprocess_dict["det_path"] = args.yolo_det
        preprocess_dict["det_extractor"] = ExtractorYOLO()
        if args.xml_ann is not None:
            preprocess_dict["ann_path"] = args.xml_ann
            preprocess_dict["ann_extractor"] = ExtractorXML()
        elif args.json_ann is not None:
            preprocess_dict["ann_path"] = args.json_ann
            preprocess_dict["ann_extractor"] = ExtractorJSON()
        elif args.coco_ann is not None:
            preprocess_dict["ann_path"] = args.coco_ann
            preprocess_dict["ann_extractor"] = ExtractorCOCO()
        else:
            raise Exception("The specified directory does not exist before files with annotations")
        preprocess = YOLOPreprocessor(preprocess_dict).preprocess()

    if args.ed_det is not None:
        if not os.path.exists(args.yolo_det):
            raise Exception("The specified directory does not exist before files with detections")
        preprocess_dict["det_path"] = args.ed_det
        preprocess_dict["det_extractor"] = ExtractorED()
        if args.xml_ann is not None:
            preprocess_dict["ann_path"] = args.xml_ann
            preprocess_dict["ann_extractor"] = ExtractorXML()
        elif args.json_ann is not None:
            preprocess_dict["ann_path"] = args.json_ann
            preprocess_dict["ann_extractor"] = ExtractorJSON()
        elif args.coco_ann is not None:
            preprocess_dict["ann_path"] = args.coco_ann
            preprocess_dict["ann_extractor"] = ExtractorCOCO()
        else:
            raise Exception("The specified directory does not exist before files with annotations")
        preprocess = EfficientDetPreprocessor(preprocess_dict).preprocess()

    metrics = calculate_metrics(preprocess)
    data_for_figure = {
        "data": metrics,
        "classes": preprocess_dict["classes"],
        "save": None,
        "show": None,
    }

    if args.show_fig:
        data_for_figure["show"] = True

    if args.save_fig:
        data_for_figure["save"] = True

    figure(data_for_figure)

    if args.show_term:
        Writer(data_for_figure).write_to_terminal()

    if args.save_metric:
        Writer(data_for_figure).write_to_csv()


if __name__ == "__main__":
    main()
