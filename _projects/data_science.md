---
layout: project
title: Data Science and ML Projects
caption: Any data science and machine learning projects I have attempted
description: >
  Over time, I have attempted many different machine learning projects to utilize as many different
  architectures as I can. Generally these projects are all deep learning based, and so some examples include
  image recognition and NLP.
date: 1 Jan 2017
image:
  path: /assets/img/projects/ml_img_medium.jpg
  srcset:
    640w: /assets/img/projects/ml_img_small.jpg
    1920w: /assets/img/projects/ml_img_medium.jpg
    2400w: /assets/img/projects/ml_img_large.jpg
    4069w: /assets/img/projects/ml_img_orig.jpg
links:
  - title: Github
    url: https://github.com/MughilM
---

This page mirrors what is currently present in my [main data science repository](https://github.com/MughilM) (also 
linked above). These follow a framework for training that is adapted from the 
[this repository](https://github.com/ashleve/lightning-hydra-template/tree/main) by ashleve. A lot of credit goes 
to this person for providing me the starter template. Below you will find the projects that have been started and
built using this template. These experiments are also capable of outputting metrics to Weights and Biases. Finally,
most of these projects are derived from the Kaggle competition counterparts.

## Histopathologic Cancer Detection
This was the "intro" project I did in order to familiarize myself with the template and how it works.
It is derived from [the corresponding Kaggle competition](https://www.kaggle.com/c/histopathologic-cancer-detection)
from 5 years ago.
It is a basic convolutional neural net that attempts to detect cancer in images.

## Oxford Pet detection
The [Oxford-IIIT Pet Dataset] is a collection of 37 categories with about 200 images per class.
It consists of both object detection (predicting bounding boxes around objects) and segmentation (predicting
each pixel individually). 
I used this dataset to primarily practice building models for image segmentation.

## Google Research - Detecting Contrails in the Atmosphere
This is a more involved project which involved predicting the exact presence of contrail in the atmosphere
given a time-series of infrared images. This is a segmentation task, where a prediction is made for each 
individual pixel in the image.

Photo by <a href="https://unsplash.com/@choys_?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Conny Schneider</a> on <a href="https://unsplash.com/photos/a-blue-abstract-background-with-lines-and-dots-pREq0ns_p_E?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Unsplash</a>
{:.faded}