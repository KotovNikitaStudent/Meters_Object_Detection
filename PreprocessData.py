import os
from collections import defaultdict


class Preprocessor:
    def __init__(self, data):
        self.classes = data['classes']
        self.ann_extractor = data['ann_extractor']
        self.det_extractor = data['det_extractor']
        self.ann_path = data['ann_path']
        self.det_path = data['det_path']
        self.names_test_samples = data['names_test_samples']

    def preprocess(self):
        pass


class EfficientDetPreprocessor(Preprocessor):
    def __init__(self, data):
        super(EfficientDetPreprocessor, self).__init__(data)

    def preprocess(self):
        all_ann, all_det = [], []

        all_samples = get_list_of_images(self.names_test_samples)
        full_path_to_ann = test_files(sorted(os.listdir(self.ann_path)), all_samples)
        full_path_to_det = test_files(sorted(os.listdir(self.det_path)), all_samples)

        for anf, dtf in zip(full_path_to_ann, full_path_to_det):
            loc_path_to_ann = os.path.join(self.ann_path, anf)
            loc_path_to_det = os.path.join(self.det_path, dtf)

            temp_dict_ann, temp_dict_det = {}, {}

            for cl in self.classes:
                temp_dict_ann.update({cl: []})
                temp_dict_det.update({cl: []})

            for i in self.ann_extractor.extract(loc_path_to_ann):
                temp_dict_ann[i[1]].append([i[2], i[3], i[4], i[5]])

            all_ann.append(temp_dict_ann)

            for i in self.det_extractor.extract(loc_path_to_det):
                temp_dict_det[i[1]].append([i[2], i[3], i[4], i[5], i[6]])

            all_det.append(temp_dict_det)

        return {'annotations': all_ann,
                'detections': all_det}


class YOLOPreprocessor(Preprocessor):
    def __init__(self, data):
        super(YOLOPreprocessor, self).__init__(data)

    def preprocess(self):
        all_ann, all_det = [], []

        # Annotation preprocessing

        all_samples = get_list_of_images(self.names_test_samples)
        full_path_to_ann = test_files(sorted(os.listdir(self.ann_path)), all_samples)

        for anf in full_path_to_ann:
            loc_path_to_ann = os.path.join(self.ann_path, anf)

            temp_dict_ann = {}
            for cl in self.classes:
                temp_dict_ann.update({cl: []})

            for i in self.ann_extractor.extract(loc_path_to_ann):
                temp_dict_ann[i[1]].append([i[2], i[3], i[4], i[5]])

            all_ann.append(temp_dict_ann)

        # Detection preprocessing

        concat_res = []
        for dtf in sorted(os.listdir(self.det_path)):
            loc_path_to_det = os.path.join(self.det_path, dtf)
            det_extract = self.det_extractor.extract(loc_path_to_det)
            det_extract.sort(key=lambda x: x[0])

            for j in det_extract:
                concat_res.append(j)

        d = defaultdict(list)

        for k, v in [(i[0], i[1:]) for i in concat_res]:
            d[k].append(v)

        for smp in sorted(d.items()):
            temp_dict_det = {}
            for cl in self.classes:
                temp_dict_det.update({cl: []})

            for ex in smp[1]:
                temp_dict_det[ex[0]].append(ex[1:])

            all_det.append(temp_dict_det)

        return {'annotations': all_ann,
                'detections': all_det}


def get_list_of_images(path_to_file: str) -> list:
    if not path_to_file.split('/')[-1].endswith('.txt'):
        raise Exception("Wrong file extension")
    else:
        list_of_files = []
        with open(path_to_file, 'r') as file:
            for line in file:
                list_of_files.append(line.strip())
        file.close()

        return sorted(list_of_files)


def test_files(curr_files, test_list):
    list_files = []

    for cf in curr_files:
        if cf.split('.')[0] in test_list:
            list_files.append(cf)

    return list_files