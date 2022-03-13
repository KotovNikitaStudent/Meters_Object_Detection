def extract_class_names_from_file(path_to_file: str) -> list:
    """Extract names of classes from file
    :param path_to_file: absolut path to file with list of class names
    :return: list of names of classes
    """
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
