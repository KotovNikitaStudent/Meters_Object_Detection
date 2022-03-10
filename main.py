from Extract_Annotation import *
from Extract_Detection import *
from PreprocessData import EfficientDetPreprocessor, YOLOPreprocessor
from calculate_metrics import calculate_metrics
from figure import figure
from Writer import Writer
from parse_arguments import parse_arguments


def extract_class_names_from_file(path_to_file: str) -> list:
    """Extract names of classes from file"""
    import json

    classes = None
    if path_to_file.endswith(".json"):
        classes = sorted(
            [class_["name"] for class_ in json.load(open(path_to_file, "r")).values()]
        )
    if path_to_file.endswith(".names"):
        with open(path_to_file, "r") as f:
            classes = sorted(f.read().split("\n"))
        f.close()

    return classes


def main():
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
        preprocess_dict["det_path"] = args.yolo_det
        preprocess_dict["det_extractor"] = ExtractorYOLO()
        if args.xml_ann is not None:
            preprocess_dict["ann_path"] = args.xml_ann
            preprocess_dict["ann_extractor"] = ExtractorXML()
        if args.json_ann is not None:
            preprocess_dict["ann_path"] = args.json_ann
            preprocess_dict["ann_extractor"] = ExtractorJSON()
        if args.coco_ann is not None:
            preprocess_dict["ann_path"] = args.coco_ann
            preprocess_dict["ann_extractor"] = ExtractorCOCO()
        preprocess = YOLOPreprocessor(preprocess_dict).preprocess()

    if args.ed_det is not None:
        preprocess_dict["det_path"] = args.ed_det
        preprocess_dict["det_extractor"] = ExtractorED()
        if args.xml_ann is not None:
            preprocess_dict["ann_path"] = args.xml_ann
            preprocess_dict["ann_extractor"] = ExtractorXML()
        if args.json_ann is not None:
            preprocess_dict["ann_path"] = args.json_ann
            preprocess_dict["ann_extractor"] = ExtractorJSON()
        if args.coco_ann is not None:
            preprocess_dict["ann_path"] = args.coco_ann
            preprocess_dict["ann_extractor"] = ExtractorCOCO()
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
