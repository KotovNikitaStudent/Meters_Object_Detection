import matplotlib.pyplot as plt


def figure(data):
    precision = data["pr"]
    recall = data["rc"]
    score = data["sc"]
    f1 = data["f1_m"]

    class_name = [
        "breaker",
        "mag",
        "meter",
        "model",
        "seal",
        "seal2",
        "serial",
        "value",
    ]

    for p, r, class_ in zip(precision, recall, class_name):
        plt.plot(r, p, label=class_)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision vs. Recall curve")
        plt.grid(True)
    plt.legend()
    plt.show()
    plt.close()

    for f, sc, class_ in zip(f1, score, class_name):
        plt.plot(sc, f, label=class_)
        plt.xlabel("Threshold")
        plt.ylabel("F1")
        plt.title("F1(Threshold) curve")
        plt.grid(True)
    plt.legend()
    plt.show()
    plt.close()
