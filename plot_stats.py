import matplotlib.pyplot as plt
import numpy as np


def plot_loss():
    train_loss = [0.4236, 0.3027, 0.2692, 0.2506, 0.2361, 0.2061, 0.1952, 0.1868, 0.1772, 0.1759, 0.1670, 0.1684, 0.1674, 0.1647, 0.1608, 0.1606, 0.1581, 0.1584, 0.1600, 0.1601]
    test_loss = [0.3242, 0.2624, 0.2510, 0.2389, 0.2255, 0.1884, 0.1808, 0.1806, 0.1740, 0.1713, 0.1681, 0.1660, 0.1647, 0.1660, 0.1639, 0.1643, 0.1602, 0.1623, 0.1594, 0.1581]
    epoch = list(range(1, 21))
    plt.plot(epoch, train_loss, 'r', label="Train Loss")
    plt.plot(epoch, test_loss, 'b', label = "Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Triplet Loss for transfer training on masked LFW")
    plt.legend(loc="upper right")
    plt.show()

def plot_precision_recall():
    precision = [0.9736390578692872, 0.9582747716092762, 0.942850174782538, 0.9289702233250621, 0.9132986877032938, 0.900076472087688, 0.8855387456495489, 0.8706486655329192, 0.8554895638893603, 0.8404785643070788]
    recall = [0.7289501342987271, 0.8493129354976838, 0.9029545719957959, 0.9326949277901048, 0.9509128420724824, 0.9621627934135233, 0.9706489158783915, 0.9765269181361672, 0.9812371053758417, 0.9844680602592549]
    precision_2 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9997337770382696, 0.9990694109918984, 0.9978741907430669, 0.9956]
    recall_2 = [0.0007396161781307174, 0.013858071548133443, 0.06029818210128849, 0.1532951847094087, 0.2833119233913348, 0.4369963797734439, 0.5847249795632372, 0.7104597298454591,0.8040017127953599, 0.8720074740161159]
    plt.plot(recall, precision, 'r', label="Transfer Trained Model")
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall Curve for transfer learning model on masked LFW")
    plt.show()

    plt.plot(recall_2, precision_2, 'b', label="Base Pretrained Model")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall Curve for base learning model on masked LFW")
    plt.show()
    


def plot_accuracy():
    thresholds = np.arange(0.1, 1.1, 0.1)
    accuracy = [0.8546070302464089, 0.906166063295574, 0.9241114874070614, 0.9306901786756978, 0.9303203705866324, 0.9276733232122698, 0.9225933278835299, 0.9157226828603683, 0.9077426135700105, 0.8988088286815369]
    accuracy_1 = [0.5003698080890654,  0.5069290357740667, 0.5301490910506442, 0.5766475923547043, 0.6416559616956674,  0.718498189886722, 0.7922846354470785, 0.8548989840009342, 0.9011444587177391, 0.9340768422281911]
    
    plt.plot(thresholds, accuracy, 'r', label="Transfer Trained Model")
    plt.plot(thresholds, accuracy_1, 'b', label="Base Pretrained Model")
    plt.xlabel("Distance Threshold")
    plt.ylabel("Accuracy")
    plt.title("Accuracy of trained model on masked LFW")
    plt.legend(loc="upper right")
    plt.show()

if __name__ == '__main__':
    # plot_loss()
    # plot_precision_recall()
    plot_accuracy()