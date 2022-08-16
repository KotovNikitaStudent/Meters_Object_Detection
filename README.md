# Meters Object Detection

F1-score, Recall, Precision, mAP, AP, AUC metrics are calculated.
According to the files with detections and annotations, dependency curves Precision(Recall), F1-score(Threadshold), ROC-curve are built for each class.
Metric values can be output to the terminal, saved to a file with the `.csv` extension in the `results` folder. The name of the file with metrics consists of a prefix, date and time of creation.
It is possible to save the figures to a file with the extension `.jpg` in the folder `results/figures`. The name of each figure consists of a prefix belonging to the type of chart, date and time of creation.

# Requirements and environment

### Linux/MacOS

Create a directory for the project. Go to the project directory and create a virtual environment for python in it (where `name_of_venv` is the name of your virtual environment):

```console
virtualenv name_of_venv
```

Starting the virtual environment:

```console
source name_of_venv/bin/activate
```

Install required packages:

```console
pip3 install -r requirements.txt
```

# Run service

The program is launched using the `run.sh` file:

```console
source run.sh
```

The script can be modified. An example of the current `run.sh` script:

```console
python3 main.py --test_files /path_to_file_with_names_of_test_images/test.txt \
--cls_names /path_to_file_with_names_of_classes/classes.json \
--yolo_det /path_to_files_with_yolo_detections/ \
--xml_ann /path_to_files_with_yolo_annotations/ \
--show_fig \
--save_fig \
--show_term \
--save_metric
```

Flags (MANDATORY):

- `--test_files` - absolute path to the file with the names of all test images
- `--cls_names` - absolute path to a file with the names of all classes (in the format or `.names`)
- `--yolo_det` - absolute path to all files with detections in `.txt` format (number of files
  along this path coincides with the number of studied classes)
- `--yolo_det` - absolute path to all files with detections in `.txt` format (number of files
  along this path coincides with the number of studied classes)
- `--ed_det` - absolute path to all files with detections in `.txt` format (number of files
  along this path coincides with the number of studied test images)
- `--xml_ann` - absolute path to all files with annotations in `.xml` format
- `--json_ann` - absolute path to all files with annotations in `.json` format (number of files
  along this path coincides with the number of studied test images)

Flags (OPTIONAL):

- `--show_fig` - show all images
- `--save_fig` - save all images
- `--show_term` - show calculated metrics in terminal
- `--save_metric` - save all metrics to a file

# Result

It is possible to display drawings on the screen or save them. The calculation of the metrics is output to the terminal or saved to a `.csv` file in the `results` folder.

<table width="1000" margin=auto>
  <td><img src="results/figures/F1_2022-03-13_16:59:27.jpg" width="213" height="160"></td>
  <td><img src="results/figures/RP_2022-03-13_16:59:25.jpg" width="213" height="160"></td>
  <td><img src="results/figures/terminal_output.jpg" width="213" height="160"></td>
</table>
