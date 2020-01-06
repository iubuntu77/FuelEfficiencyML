from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
#import tensorflow_docs as tfdocs
#import tensorflow_docs.modeling
#import tensorflow_docs.plots
from tensorflow import keras
from tensorflow.keras import layers
from google.cloud import storage

print(tf.__version__)

#keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

def preprocess(input_file, output_folder, bucket_name):

    input_file = bucket_name + '/' + input_file

    column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                    'Acceleration', 'Model Year', 'Origin']

    raw_dataset = pd.read_csv(input_file, names=column_names,
                          na_values = "?", comment='\t',
                          sep=" ", skipinitialspace=True)

    dataset = raw_dataset.copy()
    dataset = dataset.dropna()
    dataset['Origin'] = dataset['Origin'].map(lambda x: {1: 'USA', 2: 'Europe', 3: 'Japan'}.get(x))
    dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')

    train_dataset = dataset.sample(frac=0.8,random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    train_stats = train_dataset.describe()
    train_stats.pop("MPG")
    train_stats = train_stats.transpose()
    train_stats

    train_labels = train_dataset.pop('MPG').to_frame()
    test_labels = test_dataset.pop('MPG').to_frame()

    normed_train_data = norm(train_dataset, train_stats)
    normed_test_data = norm(test_dataset, train_stats)

    train_output_file = bucket_name + '/' + output_folder + '/train.csv'
    train_label_file  = bucket_name + '/' + output_folder + '/train_label.csv'
    test_output_file = bucket_name + '/' + output_folder + '/test.csv'
    test_label_file = bucket_name + '/' + output_folder + '/test_label.csv'

    print('About to write the training data') 
    print(normed_train_data)

    normed_train_data.to_csv(train_output_file)
    normed_test_data.to_csv(test_output_file)
    train_labels.to_csv(train_label_file)
    test_labels.to_csv(test_label_file)

    normed_train_data.to_csv("train.csv")

    # uploadToGCS(normed_train_data, "train.csv", bucket_name)
    # uploadToGCS(normed_test_data, "test.csv", bucket_name)



def norm(x, train_stats):
  return (x - train_stats['mean']) / train_stats['std']


def uploadToGCS(df, fileName, bucket_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(fileName)
    blob.upload_from_string(df.to_csv(), 'text/csv')

    print(
        "File {} uploaded to {}.".format(
            fileName, bucket
        )
    )

if __name__ == '__main__':
    if len(sys.argv) != 4:
       print("Usage:preprocess input_file output_folder  bucket-name")
       sys.exit(-1)
    input_file = sys.argv[1]
    output_folder = sys.argv[2]
    bucket_name = sys.argv[3]
    preprocess(input_file, output_folder, bucket_name)
