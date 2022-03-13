import datetime
import math
import os

import numpy as np
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
from beautifultable import BeautifulTable
from termcolor import colored
import pandas as pd


class Writer:
    """Write calculation results to terminal or .csv file
    Attributes
        ----------
        data : dict
            Dictionary with data for writing metrics with keys:
            "pr": list (list of precisions for each classes)
            "rc": list (list of recall for each classes)
            "sc": list (list of scores for each classes)
            "f1_m": list (list of F1-score for each classes)
            "ap": list (list of AP for each classes)
            "classes": list (list of names of classes)
    Methods
        ----------
        write_to_terminal()
            Print metrics for each class to the terminal
        write_to_csv()
            Print metrics for each class to the .csv file
        __truncate(number, digits=4)
            Trim number to n-th decimal place
    """

    def __init__(self, data):
        self._precision = data["data"]["pr"]
        self._recall = data["data"]["rc"]
        self._score = data["data"]["sc"]
        self._f1 = data["data"]["f1_m"]
        self._ap = data["data"]["ap"]
        self._classes = data["classes"]

    def write_to_terminal(self) -> None:
        """
        Print metrics for each class to the terminal
        """
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

    def write_to_csv(self) -> None:
        """
        Print metrics for each class to the .csv file
        """
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

        if not os.path.exists("./results"):
            os.mkdir("./results")

        frame.to_csv(
            f"results/meters_{datetime.datetime.today().strftime('%Y-%m-%d_%H:%M:%S')}.csv",
            sep=";",
            index=False,
            index_label=True,
        )

    @classmethod
    def __truncate(cls, number: float, digits=4) -> float:
        """
        Trim number to nth decimal place
        :param number: floating point number
        :param digits: the digit to which the number is to be truncated
        :return: truncated number
        """
        stepper = 10.0**digits
        return math.trunc(stepper * number) / stepper
