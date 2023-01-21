#import necessary library
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import random,os
import shutil
import matplotlib.pyplot as plt
from matplotlib.image import imread
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import categorical_accuracy
from sklearn.model_selection import train_test_split

print(tf.__version__)

def Generate_new_feature_in_csv(input=None) -> None:

    '''Generate new feature, Mapping the output feature'''

    if input == None:
        input = r'/home/vk/Desktop/Retina_Webapp/archive/train.csv'
    
    data = pd.read_csv(input)
    print(data.head())

    Defect_binary = {
        0:'No_DR',
        1: 'DR',
        2: 'DR',
        3: 'DR',
        4: 'DR'
    }

    diagnosis_all_dict = {
        0:'No_DR',
        1: 'Mild',
        2: 'Moderate',
        3: 'Severe',
        4: 'Proliferate_DR',
    }

    data['binary_type'] = data['diagnosis'].map(Defect_binary.get)
    data['type'] = data['diagnosis'].map(diagnosis_all_dict.get)

    print(data.head())

    data['type'].value_counts().plot(kind='hist')



Generate_new_feature_in_csv()
print(Generate_new_feature_in_csv.__doc__)
