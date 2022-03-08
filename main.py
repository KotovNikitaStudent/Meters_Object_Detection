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
    class_names = None
    detection_extractor = None
    annotation_extractor = None
    path_to_ann = None
    path_to_det = None
    names_test_samples = None

    args = parse_arguments()

    if args.test_files is not None:
        names_test_samples = args.test_files
    else:
        raise Exception('You did not specify the absolute path to the file with all test instances')

    if args.cls_names is not None:
        class_names = args.cls_names
    else:
        raise Exception('You did not specify an absolute path to the file with class names')

    if args.xml_ann is not None:
        path_to_ann = args.xml_ann
        annotation_extractor = ExtractorXML()
    else:
        raise Exception('Annotation file path does not exist')

    if args.json_ann is not None:
        path_to_ann = args.json_ann
        annotation_extractor = ExtractorJSON()
    else:
        raise Exception('Annotation file path does not exist')

    if args.coco_ann is not None:
        path_to_ann = args.coco_ann
        annotation_extractor = ExtractorCOCO()
    else:
        raise Exception('Annotation file path does not exist')

    if args.ed_det is not None:
        path_to_ann = args.ed_det
        detection_extractor = ExtractorED()
    else:
        raise Exception('Detection file path does not exist')

    if args.yolo_det is not None:
        path_to_ann = args.yolo_det
        detection_extractor = ExtractorYOLO()
    else:
        raise Exception('Detection file path does not exist')

    preprocess_dict = {
        "classes": class_names,
        "ann_extractor": annotation_extractor,
        "det_extractor": detection_extractor,
        "ann_path": path_to_ann,
        "det_path": path_to_det,
        "names_test_samples": names_test_samples,
    }

    preprocess = None
    if args.yolo:
        preprocess = YOLOPreprocessor(preprocess_dict).preprocess()
    else:
        raise Exception("You didn't specify a required parameter '--yolo' for input preprocessing")

    if args.efficient_det:
        preprocess = EfficientDetPreprocessor(preprocess_dict).preprocess()
    else:
        raise Exception("You didn't specify a required parameter '--efficient_det' for input preprocessing")

    metrics = calculate_metrics(preprocess)

    data_for_figure = {"data": metrics,
             "classes": class_names,
             "save": False,
             "show": False,
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
