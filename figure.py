import matplotlib.pyplot as plt


def get_metrics_and_figure(data):
    precision = data['pr']
    recall = data['rc']
    score = data['sc']
    f1 = data['f1_m']
    # ap = data['ap']

    class_name = ['breaker', 'mag', 'meter', 'model', 'seal', 'seal2', 'serial', 'value']
    colors = ['purple', 'sienna', 'green', 'orange', 'gray', 'r', 'violet', 'blue']

    for p, r, class_, col in zip(precision, recall, class_name, colors):
        plt.plot(r, p, label=class_, color=col)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision vs. Recall curve')
        plt.grid(True)
    plt.legend()
    plt.show()

    for f, sc, class_, col in zip(f1, score, class_name, colors):
        plt.plot(sc, f, label=class_, color=col)
        plt.xlabel('Threshold')
        plt.ylabel('F1')
        plt.title('F1(Threshold) curve')
        plt.grid(True)
    plt.show()
    plt.legend()