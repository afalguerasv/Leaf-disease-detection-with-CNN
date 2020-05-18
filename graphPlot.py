import matplotlib.pyplot as plt

def plotGraph(acc, val_acc, loss, val_loss, epochs_range, titleAcc, titleLoss, min, max):
    plt.subplot(2, 1, 2)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(titleAcc)
    plt.xticks(range(min, max))
    plt.grid()

    plt.subplot(2, 1, 1)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(titleLoss)
    plt.xticks(range(min, max))
    plt.grid()
    plt.show()



