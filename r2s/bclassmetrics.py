from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def confusion_matrix_metrics(y_true, y_pred):
    try:
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        labels = []
        false_positive_rate = fp / (fp + tn)
        labels.append(f'False positive rate: {false_positive_rate}')
        false_negative_rate = fn / (tp + fn)
        labels.append(f'False negative rate: {false_negative_rate}')

        fig, ax = plt.subplots()
        cmap = plt.get_cmap('Blues')

        sns.heatmap(cm, cmap=cmap, annot=True, fmt='g', ax=ax)
        plt.xlabel('predicted values')
        plt.ylabel('actual values')
        ax.legend(title=None, loc='upper left', labels=labels)
        plt.show()
    except Exception as msg:
        breakpoint()
        exit(msg)
