import os
import datetime
import matplotlib.pyplot as plt


def figure(data):
    precision = data["data"]["pr"]
    recall = data["data"]["rc"]
    score = data["data"]["sc"]
    f1 = data["data"]["f1_m"]
    class_name = data["classes"]

    tp_roc = data["data"]["tp_roc"]
    fp_roc = data["data"]["fp_roc"]

    if not os.path.exists("./figures"):
        os.mkdir("./figures")

    plt.figure(1)
    for p, r, class_ in zip(precision, recall, class_name):
        plt.plot(r, p, label=class_)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision vs. Recall curve")
        plt.grid(True)
    plt.legend()
    if data["save"]:
        plt.savefig(
            f"figures/RP_{datetime.datetime.today().strftime('%Y-%m-%d_%H:%M:%S')}.jpg"
        )
    if data["show"]:
        plt.show()

    plt.figure(2)
    for f, sc, class_ in zip(f1, score, class_name):
        plt.plot(sc, f, label=class_)
        plt.xlabel("Threshold")
        plt.ylabel("F1")
        plt.title("F1(Threshold) curve")
        plt.grid(True)
    plt.legend()
    if data["save"]:
        plt.savefig(
            f"figures/F1_{datetime.datetime.today().strftime('%Y-%m-%d_%H:%M:%S')}.jpg"
        )
    if data["show"]:
        plt.show()

    plt.figure(3)
    for scr, fp_rc, tp_rc, class_ in zip(score, fp_roc, tp_roc, class_name):
        x, y, auc = curve_roc(scr, tp_rc, fp_rc)
        plt.plot(x, y, label=class_ + f" (AUC = {round(auc, 3)})")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC curve")
        plt.grid(True)
    plt.legend()
    if data["save"]:
        plt.savefig(
            f"figures/ROC_{datetime.datetime.today().strftime('%Y-%m-%d_%H:%M:%S')}.jpg"
        )
    if data["show"]:
        plt.show()


def curve_roc(score, fp_roc, tp_roc):
    db = []
    pos, neg = 0, 0
    for scr, fp_rc, tp_rc in zip(score, fp_roc, tp_roc):
        fp_rc = int(fp_rc)
        tp_rc = int(tp_rc)
        scr = float(scr)
        db.append([scr, fp_rc, tp_rc])
        pos += tp_rc
        neg += fp_rc

    db = sorted(db, key=lambda x: x[0], reverse=True)

    xy_arr = []
    tp, fp = 0.0, 0.0
    for i in range(len(db)):
        tp += db[i][2]
        fp += db[i][1]
        xy_arr.append([fp / neg, tp / pos])

    auc = 0.0
    prev_x = 0
    for x, y in xy_arr:
        if x != prev_x:
            auc += (x - prev_x) * y
            prev_x = x

    x = [_v[0] for _v in xy_arr]
    y = [_v[1] for _v in xy_arr]

    return [x, y, auc]
