---
layout: project
title: Wildfire Incidence Prediction (CU)
caption: Predicting incidence of wildfires across the US as part of a class at CU
description: >
  This final project was done as part of the Data Science and Public Policy class in Columbia University along
  with other students.
date: 13 May 2020
image:
  path: /assets/img/projects/wildfire_medium.jpg
  srcset:
    640w: /assets/img/projects/wildfire_small.jpg
    1920w: /assets/img/projects/wildfire_medium.jpg
    2400w: /assets/img/projects/wildfire_large.jpg
    7952w: /assets/img/projects/wildfire_orig.jpg
links:
  - title: Github (incl. report)
    url: https://github.com/MughilM/AcaDS/tree/main/INAF6506/Final%20project
  - title: Presentation
    url: https://docs.google.com/presentation/d/1zgRDMos9obiZm_COSr4ImQ-sVFeh-ebA/edit?usp=sharing&ouid=112620148513863692881&rtpof=true&sd=true
---

This data science project was done as part of the **Data Science and Public Policy** class at
Columbia University, taught by Prof. Tamar Mitts. The class took a public policy approach to various
data science problems, with a focus on developing real-world solutions. The final task was to take
any actionable problem and apply any data science/ML techniques necessary and compile a final report.
The presentation and report was done in collaboration with 3 other students: Max Mauerman, Priyanka Sethy,
and Pei Yin Teo, who contributed heavily in various areas.

The general idea of this project was to attempt to predict incidence of wildfires given past wildfire
incidence data, along with land usage and weather data. 
The wildfire incidence data is from the USDA[^2], and consists of approximately 1.88 million wildfires that have happened 
in the US from 1992 to 2015. For historical climate, we used WorldClim[^1].
For land cover data, we use the [National Land Cover Database](https://www.usgs.gov/centers/eros/science/national-land-cover-database#overview)
from the USGS.
For additional details concerning exactly which data and how we processed, please consult the report.

In terms of machine learning methods used, we decided to use spatial logistic regression. We break down the
contiguous United States into cells, and predict fire incidence in a particular cell. For prediction, we utilize
the above data in surrounding cells as input, and output a value between 0 and 1 for the current cell. Finally,
we also attempted to use basic neural networks to perform the same task.

The following blog posts will go over the code in detail, including preprocessing and training.
These correspond with the notebooks that are in the Github repository folder.


Photo by <a href="https://unsplash.com/@mattpalmer?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Matt Palmer</a> on <a href="https://unsplash.com/photos/silhouette-of-trees-during-sunset-kbTp7dBzHyY?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Unsplash</a>
{:.faded}

[^1]: Fick, S.E. and R.J. Hijmans, 2017. WorldClim 2: new 1km spatial resolution climate surfaces for global land areas. [International Journal of Climatology 37 (12): 4302-4315](https://rmets.onlinelibrary.wiley.com/doi/abs/10.1002/joc.5086)
[^2]: Short, Karen C. 2017. Spatial wildfire occurrence data for the United States, 1992-2015 [FPA_FOD_20170508]. 4th Edition. Fort Collins, CO: Forest Service Research Data Archive. https://doi.org/10.2737/RDS-2013-0009.4
.