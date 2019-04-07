# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 18:56:47 2019

@author: cidm
"""

def image_show(some_digit):
    import matplotlib.pyplot as plt
    import matplotlib
    some_digit_image = some_digit.reshape(28,28)
    plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation='nearest')
    plt.axis('off')
    return plt.show()

def plot_precision_recall_threshold(precisions, recalls, thresholds):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot (thresholds, recalls[:-1], 'g-', label='Recall')
    plt.xlabel('Threshold')
    plt.xticks(np.arange(min(thresholds), max(thresholds)+150000, 150000))
    plt.legend(loc='center left')
    plt.ylim([0,1])
    return plt.show()

def plot_precision_recall(precisions, recalls):
    import matplotlib.pyplot as plt
    plt.plot (recalls, precisions, 'g-')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    return plt.show()

def plot_roc_curve(fpr, tpr, label=None):
    import matplotlib.pyplot as plt
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1], [0,1],'k--')#represents a purely random classifier for reference
    plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    return plt.show()
    