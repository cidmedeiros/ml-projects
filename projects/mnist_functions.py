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