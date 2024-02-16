---
layout: post
title: Wildfire Incidence III - Basic Neural Network
description: >
  Part 3 of predicting wildfire incidence. I show a basic example of using
  a neural network on the same data for prediction.
date: 19 May 2020
image:
  path: /assets/img/projects/wildfire_medium.jpg
  srcset:
    640w: /assets/img/projects/wildfire_small.jpg
    1920w: /assets/img/projects/wildfire_medium.jpg
    2400w: /assets/img/projects/wildfire_large.jpg
    7952w: /assets/img/projects/wildfire_orig.jpg
accent_color: '#e25822'
theme_color: '#e25822'
related_posts:
  - wildfires/_posts/2020-05-15-wildfire-preprocessing.md
  - wildfires/_posts/2020-05-17-wildfire-regression.md
tags: [geospatial data, neural network, climate]
---

# Wildfire Incidence III - Basic Neural Network

We have seen how to properly preprocess the 3 data sources (historical wildfire incidence, climatology,
and land usage) to get it ready to push it through a logistic regression model. However, the beauty of machine
learning is that there is often more than one tool for the job.
{:.lead}

* toc
{:toc}


## Introduction
The previous notebook saved the post-processed data into a CSV so we can easily read it in this notebook.
Due to the relatively limited number of input variables, the network will not be very big and computationally
intensive. Moreover, the network structure allows for non-linear relationships to be built *between* inputs.

It should be noted that this input format is not very well suited to the neural network structure, but we
will press forward regardless.

## Import Packages
As always, we import all the packages we will use. To build the network, we will be using **Tensorflow**.
However, there is one caveat here if you are using Windows. If using native Windows, then the maximum allowed
Python version is 3.10. This is due to Tensorflow 2.10 being the last version **that allows for native GPU
support.** If newer versions of Tensorflow wish to be used, then WSL2 need to be used instead.
However, using Python 3.10 does not influence the functionality of the other required packages in any way.

```python
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import make_column_transformer
from sklearn.metrics import classification_report

import tensorflow.keras.layers as layers
from tensorflow.keras.models import Sequential

# IMPORTANT, this is the same seed as the logistic regression, 
# so the same train test split is generated, and metrics are comparable.
np.random.seed(7145)
ROOT = '../'
```
Notice we used the same seed as the logistic regression. When we perform a train test split, we want to ensure
the same training and testing set is produced for a much cleaner comparison.

## Metadata and Data Reading
This should be familiar by now, but we will create variables for directory locations, and other convenient
variables. Importantly, we read in the **spatialReg** CSV file that was generated during the logistic
regression notebook after all the lagged variables were calculated. This will save us a lot of time and code.
```python
LAG = 5
RES = '10m'
PREFIX = os.path.join(ROOT, 'data')
PROCESSED_PATH = os.path.join(PREFIX, 'processed', RES)

spatial_reg = pd.read_csv(os.path.join(PROCESSED_PATH, 'spatialReg.csv'))
spatial_reg.head()
```

|   | Month | fires\_L1\_% | tavg\_L1\_avg | prec_L1_avg | srad_L1_avg | wind_L1_avg | vapr_L1_avg | LC11_L1_% | LC12_L1_% | LC21_L1_% | ... | LC41_L5_% | LC42_L5_% | LC43_L5_% | LC52_L5_% | LC71_L5_% | LC81_L5_% | LC82_L5_% | LC90_L5_% | LC95_L5_% | fireCenter |
|---|-------|--------------|---------------|-------------|-------------|-------------|-------------|-----------|-----------|-----------|-----|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|------------|
| 0 | Jan   | 0.0          | -17.215083    | 18.0        | 4447.666667 | 3.445333    | 0.165133    | 0.666667  | 0.0       | 0.0       | ... | 0.0       | 0.000000  | 0.0       | 0.000000  | 0.0       | 0.000000  | 0.285714  | 0.714286  | 0.000000  | 0.0        |
| 1 | Jan   | 0.0          | -17.008937    | 18.5        | 4425.500000 | 3.405188    | 0.165156    | 0.750000  | 0.0       | 0.0       | ... | 0.0       | 0.000000  | 0.0       | 0.000000  | 0.0       | 0.000000  | 0.142857  | 0.428571  | 0.428571  | 0.0        |
| 2 | Jan   | 0.0          | -16.799875    | 17.5        | 4429.000000 | 3.361625    | 0.168513    | 1.000000  | 0.0       | 0.0       | ... | 0.0       | 0.000000  | 0.0       | 0.000000  | 0.0       | 0.000000  | 0.000000  | 0.857143  | 0.142857  | 0.0        |
| 3 | Jan   | 0.0          | 4.225424      | 117.0       | 3243.000000 | 3.450847    | 0.680000    | 1.000000  | 0.0       | 0.0       | ... | 0.0       | 0.000000  | 0.0       | 0.000000  | 0.0       | 0.200000  | 0.200000  | 0.000000  | 0.000000  | 0.0        |
| 4 | Jan   | 0.0          | 3.318625      | 144.5       | 3287.000000 | 3.139708    | 0.643800    | 0.500000  | 0.0       | 0.0       | ... | 0.0       | 0.142857  | 0.0       | 0.142857  | 0.0       | 0.142857  | 0.142857  | 0.142857  | 0.000000  | 0.0        |

## Preprocessing the data
We need to follow **exactly the same processing steps** we used in the logistic regression. This involves
doing class balancing of no fires vs. fires in the training set, and appropriately transforming the climatology
metrics. Average temperature, solar radiation, and wind speeds are standardized using their mean and standard
deviation, while the precipitation and humidity are power transformed using the Yeo-Johnson method.
All standardization parameters need to calculated from the training set, and applied to the testing set.

The code block is fairly long, so for additional details, please refer to the logistic regression notebook.

```python
# First handle the multicollinearity.
for lag in range(1, LAG + 1):
    spatial_reg.drop(columns=f'LC95_L{lag}_%', inplace=True)
# Get rid of rows with NaNs in them...
spatial_reg = spatial_reg.dropna()
# Do a train test split
train_indices, test_indices = train_test_split(np.arange(spatial_reg.shape[0]), test_size=0.2)
train_set = spatial_reg.iloc[train_indices]
test_set = spatial_reg.iloc[test_indices]

# Split the input from output
X_train = train_set.iloc[:, 1:-1]
y_train = train_set['fireCenter']
X_test = test_set.iloc[:, 1:-1]
y_test = test_set['fireCenter']

# Equalize the number of positive and negative samples in the training set...
no_fire_samples = train_set[train_set['fireCenter'] == 0]
fire_samples = train_set[train_set['fireCenter'] == 1]
# Randomly choose the number of 1 samples we have from the no fire samples
chosen_no_fire_samples = no_fire_samples.sample(n=fire_samples.shape[0])
# Concatenate both sets together, and shuffle with .sample(frac=1)
train_set = pd.concat((chosen_no_fire_samples, fire_samples), axis=0, ignore_index=True).sample(frac=1)
print('New number of records:', train_set.shape[0])
print('New proportion of fires:', np.mean(train_set['fireCenter']))

# Split off X and y train again
X_train = train_set.iloc[:, 1:-1]
y_train = train_set['fireCenter']

standard_cols = [col for col in spatial_reg.columns if any(col.startswith(metric)
                                                           for metric in ['tavg', 'srad', 'wind'])]
power_cols = [col for col in spatial_reg.columns if any(col.startswith(metric)
                                                        for metric in ['prec', 'vapr'])]

transform = make_column_transformer(
    (StandardScaler(), standard_cols),
    (PowerTransformer(method='yeo-johnson'), power_cols),
    remainder='passthrough',  # To avoid dropping columns we DON'T transform
    verbose_feature_names_out=False
    # To get the final mapping of input to output columns without original transformer name.
)
transform.fit(X_train)

# Create a transformed DataFrame, with the transformed data, and the new column ordering
X_transform = pd.DataFrame(data=transform.transform(X_train),
                           columns=transform.get_feature_names_out(transform.feature_names_in_))
# Now, find the new index ordering
col_index_ordering = [X_transform.columns.get_loc(orig_col) for orig_col in X_train.columns]
# Reindexing into the column list with the new indices will automatically reorder them!
X_transform = X_transform[X_transform.columns[col_index_ordering]]

X_test_transform = pd.DataFrame(data=transform.transform(X_test),
                                columns=transform.get_feature_names_out(transform.feature_names_in_))
X_test_transform = X_test_transform[X_test_transform.columns[col_index_ordering]]
```
```text
New number of records: 140198
New proportion of fires: 0.5
```

## Building the model
We are ready to build our network. However, because we comparably do not have that many features (~100),
the network will not have too many parameters. But we can still stack a few layers to make a prediction.
Starting with 105 input features, we will have 3 layers with 75, 50, and 25 output neurons respectively.
The last layer will just consist of a single neuron for the binary prediction. We will use ReLU for
intermediate layer activations, and sigmoid for the final layer, since the output is required to be between 0
and 1.

```python
in_features = X_transform.shape[1]

model = Sequential()
model.add(layers.Dense(75, activation='relu', input_shape=(in_features,)))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(25, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  # Output!

model.summary()
```
```text
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 75)                7950      
                                                                 
 dense_1 (Dense)             (None, 50)                3800      
                                                                 
 dense_2 (Dense)             (None, 25)                1275      
                                                                 
 dense_3 (Dense)             (None, 1)                 26        
                                                                 
=================================================================
Total params: 13,051
Trainable params: 13,051
Non-trainable params: 0
_________________________________________________________________
```

## Training!
For training, we simply pass in our transformed training, and provide the transformed test data as
"validation". This way we can see the accuracy performance as it trains. We will use the standard
Adam optimizer, and the binary cross entropy loss function.

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(X_transform, y_train, batch_size=64, epochs=30, validation_data=(X_test_transform, y_test))
```
```text
Epoch 1/30
2191/2191 [==============================] - 12s 5ms/step - loss: 0.4129 - acc: 0.8122 - val_loss: 0.4090 - val_acc: 0.8127
Epoch 2/30
2191/2191 [==============================] - 12s 5ms/step - loss: 0.4034 - acc: 0.8165 - val_loss: 0.4069 - val_acc: 0.8139
Epoch 3/30
2191/2191 [==============================] - 12s 5ms/step - loss: 0.4014 - acc: 0.8172 - val_loss: 0.4147 - val_acc: 0.8102
Epoch 4/30
2191/2191 [==============================] - 10s 4ms/step - loss: 0.4004 - acc: 0.8176 - val_loss: 0.4125 - val_acc: 0.8127
Epoch 5/30
2191/2191 [==============================] - 11s 5ms/step - loss: 0.3991 - acc: 0.8183 - val_loss: 0.4064 - val_acc: 0.8131
[...]
Epoch 24/30
2191/2191 [==============================] - 11s 5ms/step - loss: 0.3864 - acc: 0.8245 - val_loss: 0.4037 - val_acc: 0.8117
Epoch 25/30
2191/2191 [==============================] - 11s 5ms/step - loss: 0.3855 - acc: 0.8244 - val_loss: 0.4050 - val_acc: 0.8112
Epoch 26/30
2191/2191 [==============================] - 11s 5ms/step - loss: 0.3845 - acc: 0.8256 - val_loss: 0.4113 - val_acc: 0.8151
Epoch 27/30
2191/2191 [==============================] - 11s 5ms/step - loss: 0.3844 - acc: 0.8253 - val_loss: 0.3950 - val_acc: 0.8144
Epoch 28/30
2191/2191 [==============================] - 11s 5ms/step - loss: 0.3832 - acc: 0.8260 - val_loss: 0.4182 - val_acc: 0.8119
Epoch 29/30
2191/2191 [==============================] - 11s 5ms/step - loss: 0.3825 - acc: 0.8263 - val_loss: 0.3987 - val_acc: 0.8138
Epoch 30/30
2191/2191 [==============================] - 11s 5ms/step - loss: 0.3819 - acc: 0.8271 - val_loss: 0.4046 - val_acc: 0.8126

<keras.callbacks.History at 0x1db91442f20>
```
On paper, the accuracy is 81%, but in our case, the accuracy doesn't tell the story.
We'll show a classification report just like the logistic regression.
## Evaluation
The evaluation process is straightforward, we grab the predictions with `model.predict`, and have our
threshold be 0.5. Any probabilities greater than this amount, and the model predicts a fire, and
anything less, no fire. This is the threshold that the logisiic regression operated under, so we'll
do as close as a comparison as we can.
```python
threshold = 0.5

predictions = model.predict(X_test_transform).ravel()
predictions[predictions < threshold] = 0
predictions[predictions >= threshold] = 1

print(classification_report(y_true=y_test, y_pred=predictions))
```
```text
2289/2289 [==============================] - 4s 2ms/step
              precision    recall  f1-score   support

           0       0.93      0.81      0.87     55761
           1       0.58      0.82      0.67     17475

    accuracy                           0.81     73236
   macro avg       0.75      0.81      0.77     73236
weighted avg       0.85      0.81      0.82     73236
```

Compared with our logistic regression, our metrics are about 1-2 points lower across the board.
Perhaps the fully connected neural network did not work as well as we thought. There are other neural
networks we can try.
## Conclusion
This relatively small post went over how to build a neural network solution to our wildfire incidence problem.
This network is extremely barebones, as it simply took our input data and fed it directly through a
fully connected network. To extract even more performance, different processing to the dataset will need
to be performed.

It might be tempting to apply an image recognition CNN network to the fire, climatology, and land usage
images themselves, but there could be reasons why a CNN may not be viable. Each pixel in the image is
treated as both an input **and** output. Conventional CNNs produce a single label given a single image.
However, here, we are basically classifying each pixel in the image.

The particular class of image problems is called **image segmentation**, which requires an entirely different
network architecture. However, the downside is that comparably, we do not have that many images to work with,
since the original climatology images were only separated by month. Additional data would need to be
collected at a more fine temporal resolution.

Neural networks should only be used as a last resort to most machine learning problems. In fact, we did
not even test very basic solutions. For example, we could have simply assigned fire incidence as such:
If a majority of neighboring cells had a fire, then the central spot had a fire. This does not require any
training, nor does it require the climatology or land usage data. This could also easily be tested at various
lag values.

So it is important to stay grounded in which solution you end up choosing, ensuring it is not overly
complicated, and that the algorithm is not being forced to solve the problem.


