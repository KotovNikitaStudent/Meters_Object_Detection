import math
import numpy as np
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
from beautifultable import BeautifulTable
from termcolor import colored
import pandas as pd


class Writer:
    def __init__(self, data):
        self._precision = data["data"]["pr"]
        self._recall = data["data"]["rc"]
        self._score = data["data"]["sc"]
        self._f1 = data["data"]["f1_m"]
        self._ap = data["data"]["ap"]
        self._classes = data["classes"]

    def write_to_terminal(self):
        table = BeautifulTable()

        table.column_headers = [
            f"{colored('Class', 'blue', attrs=['bold'])}",
            f"{colored('Precision', 'blue', attrs=['bold'])}",
            f"{colored('Recall', 'blue', attrs=['bold'])}",
            f"{colored('F1', 'blue', attrs=['bold'])}",
            f"{colored('AP', 'blue', attrs=['bold'])}",
        ]

        for i, j, k, z, cl in zip(
            self._f1, self._precision, self._recall, range(len(self._ap)), self._classes
        ):
            index = np.argmax(i)
            table.append_row(
                [
                    f"{colored(f'{cl}', 'green', attrs=['bold'])}",
                    self.__truncate(j[index]),
                    self.__truncate(k[index]),
                    self.__truncate(2 * j[index] * k[index] / (k[index] + j[index])),
                    self.__truncate(self._ap[z]),
                ]
            )
        print(table)
        print(
            f"{colored('mAP:', 'red', attrs=['bold'])} {self.__truncate(sum(self._ap) / len(self._ap))}"
        )

    def write_to_csv(self):
        precisions_list = [self.__truncate(i[np.argmax(i)]) for i in self._precision]
        recall_list = [self.__truncate(i[np.argmax(i)]) for i in self._recall]
        f1_list = [self.__truncate(i[np.argmax(i)]) for i in self._f1]
        ap_list = [self.__truncate(self._ap[i]) for i in range(len(self._ap))]

        frame = pd.DataFrame(
            {
                " ": self._classes,
                "precision": precisions_list,
                "recall": recall_list,
                "f1": f1_list,
                "ap": ap_list,
                "map": self.__truncate(sum(self._ap) / len(self._ap)),
            }
        )

        frame.to_csv(f"meters_csv.csv", sep=";", index=False, index_label=True)

    def __truncate(self, number, digits=4):
        stepper = 10.0**digits
        return math.trunc(stepper * number) / stepper
