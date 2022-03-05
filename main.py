from Extract_Annotation import *
from Extract_Detection import *
from PreprocessData import EfficientDetPreprocessor, YOLOPreprocessor
from calculate_metrics import calculate_metrics
from figure import figure


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

    class_names = sorted(
        [
            "breaker",
            "mag",
            "meter",
            "model",
            "seal",
            "seal2",
            "serial",
            "value",
        ]
    )

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
    figure(y)


if __name__ == "__main__":
    main()
