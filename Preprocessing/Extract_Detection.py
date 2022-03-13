from abc import ABC, abstractmethod


class DetectionExtractor(ABC):
    @abstractmethod
    def extract(self, path_to_file: str) -> list:
        """Extracts data for detection from file
        Arguments:
            path_to_file: str
                Absolute path to the file
        Returns:
            Data from a file with a detection
        """
        pass


class ExtractorYOLO(DetectionExtractor):
    def extract(self, path_to_file: str) -> list:
        """Extracts data for detection from .txt file. ATTENTION: each file .txt contains all detection data
        for a specific detection class
        Arguments:
            path_to_file: str
                Absolute path to the .txt file
        Returns:
            Data from a file with an annotation
            in the form [[image_name, label, x_min, y_min, x_max, y_max, score], ...]
        """
        data_from_file = []
        if path_to_file.endswith(".txt"):
            label = path_to_file.split("/")[-1].split(".")[0].split("_")[-1]
            with open(path_to_file, "r") as file:
                for line in file:
                    split_data = line.strip().split(" ")
                    image_name = split_data[0]
                    score = float(split_data[1])
                    x_min = float(split_data[2])
                    y_min = float(split_data[3])
                    x_max = float(split_data[4])
                    y_max = float(split_data[5])
                    data_from_file.append(
                        [image_name, label, x_min, y_min, x_max, y_max, score]
                    )

            return data_from_file
        else:
            print(f"The file {path_to_file} has the wrong extension")


class ExtractorED(DetectionExtractor):
    def extract(self, path_to_file: str) -> list:
        """Extracts data for detection from .txt file. ATTENTION: each file .txt contains detection data
        for a specific test image
        Arguments:
            path_to_file: str
                Absolute path to the .txt file
        Returns:
            Data from a file with an annotation
             in the form [[image_name, label, x_min, y_min, x_max, y_max, score], ...]
        """
        data_from_file = []
        if path_to_file.endswith(".txt"):
            image_name = path_to_file.split("/")[-1].split(".")[0]
            with open(path_to_file, "r") as file:
                for line in file:
                    split_data = line.strip().split(" ")
                    label = split_data[0]
                    score = float(split_data[1])
                    x_min = float(split_data[2])
                    y_min = float(split_data[3])
                    x_max = float(split_data[4])
                    y_max = float(split_data[5])
                    data_from_file.append(
                        [image_name, label, x_min, y_min, x_max, y_max, score]
                    )
            return data_from_file
        else:
            print(f"The file {path_to_file} has the wrong extension")
