from Extract_Annotation import *
from Extract_Detection import *
from PreprocessData import EfficientDetPreprocessor, YOLOPreprocessor
from calculate_metrics import calculate_metrics
from figure import figure
from Writer import Writer


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
    # EfficientDet

    # ex_xml = ExtractorXML()
    # ex_ed = ExtractorED()
    #
    # class_names = sorted(
    #     [
    #         "breaker",
    #         "mag",
    #         "meter",
    #         "model",
    #         "seal",
    #         "seal2",
    #         "serial",
    #         "value",
    #     ]
    # )
    # ANN_XML = (
    #     "/Users/nikita/Desktop/zip_meters/crop_dataset/00_dataset/meters01/Annotations/"
    # )
    # DET_ED = "/Users/nikita/Desktop/zip_meters/result_ed/ed0/05/"
    # LIST_SAMPLES = "/Users/nikita/Desktop/zip_meters/crop_dataset/00_dataset/meters01/ImageSets/Main/test.txt"
    #
    # preprocess_dict = {
    #     "classes": class_names,
    #     "ann_extractor": ex_xml,
    #     "det_extractor": ex_ed,
    #     "ann_path": ANN_XML,
    #     "det_path": DET_ED,
    #     "names_test_samples": LIST_SAMPLES,
    # }
    #
    # x = EfficientDetPreprocessor(preprocess_dict).preprocess()
    # y = calculate_metrics(x)
    # get_metrics_and_figure(y)

    # YOLO

    ex_xml = ExtractorXML()
    ex_yolo = ExtractorYOLO()

    class_names = extract_class_names_from_file("classes.json")

    ANN_XML = (
        "/Users/nikita/Desktop/zip_meters/crop_dataset/00_dataset/meters01/Annotations/"
    )
    DET_YOLO = (
        "/Users/nikita/Desktop/zip_meters/results_yolov4/results_tiny3l_meters_extra"
    )

    LIST_SAMPLES = "/Users/nikita/Desktop/zip_meters/crop_dataset/00_dataset/meters01/ImageSets/Main/test.txt"

    preprocess_dict = {
        "classes": class_names,
        "ann_extractor": ex_xml,
        "det_extractor": ex_yolo,
        "ann_path": ANN_XML,
        "det_path": DET_YOLO,
        "names_test_samples": LIST_SAMPLES,
    }

    x = YOLOPreprocessor(preprocess_dict).preprocess()
    y = calculate_metrics(x)

    data_for_figure = {"data": y, "classes": class_names, "save": False, "show": False}
    figure(data_for_figure)


if __name__ == "__main__":
    main()
