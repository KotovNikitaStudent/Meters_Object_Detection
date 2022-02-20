from abc import ABC, abstractmethod
import xml.etree.ElementTree as ET
from json import load


class AnnotationExtractor(ABC):
    @abstractmethod
    def extract(self, path_to_file: str) -> list:
        pass


class ExtractorXML(AnnotationExtractor):
    def extract(self, path_to_file: str) -> list:
        data_from_file = []
        if path_to_file.endswith('.xml'):
            root = ET.parse(path_to_file).getroot()
            image_name = root.find('filename').text[:-4]

            for obj in root.findall('object'):
                label = obj.find('name').text
                x_min = float(obj.find('bndbox/xmin').text)
                y_min = float(obj.find('bndbox/ymin').text)
                x_max = float(obj.find('bndbox/xmax').text)
                y_max = float(obj.find('bndbox/ymax').text)
                data_from_file.append([image_name, label, x_min, y_min, x_max, y_max])

            return data_from_file
        else:
            print(f"The file {path_to_file} has the wrong extension")


class ExtractorJSON(AnnotationExtractor):
    def extract(self, path_to_file: str) -> list:
        data_from_file = []
        if path_to_file.endswith('.json') or path_to_file.endswith('.JSON'):
            with open(path_to_file, 'r') as file:
                data = load(file)
            file.close()

            image_name = data['imagePath'][:-4]
            for obj in data['shapes']:
                label = obj['label']
                x_min = obj['points'][0][0]
                y_min = obj['points'][0][1]
                x_max = obj['points'][2][0]
                y_max = obj['points'][2][1]
                data_from_file.append([image_name, label, x_min, y_min, x_max, y_max])

            return data_from_file
        else:
            print(f"The file {path_to_file} has the wrong extension")


class ExtractorCOCO(AnnotationExtractor):
    def extract(self, path_to_file: str) -> list:
        data_from_file = []
        if path_to_file.endswith('.json') or path_to_file.endswith('.JSON'):
            with open(path_to_file, 'r') as file:
                data = load(file)
            file.close()

            image_name = data['images'][0]['file_name'].split('.')[0]
            class_dict = {}

            for i in data['categories']:
                class_dict.update({i['id']: i['name']})

            for i in data['annotations']:
                label = class_dict[i["category_id"]]
                center_x = i['bbox'][0]
                center_y = i['bbox'][1]
                width = i['bbox'][2]
                height = i['bbox'][3]

                x_min = center_x - (width / 2)
                x_max = center_x + (width / 2)
                y_min = center_y - (height / 2)
                y_max = center_y + (height / 2)

                data_from_file.append([image_name, label,
                                       x_min, y_min, x_max, y_max])

            return data_from_file
        else:
            print(f"The file {path_to_file} has the wrong extension")