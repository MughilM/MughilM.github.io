---
layout: project
title: Thematic Fit (CU)
caption: An NLP research project that started at Columbia University and culminated in a published paper
description: >
  As part of the final assignment at Columbia University's master's of data science program, a team and students
  and I went under the mentorship of Yuval Marton and Asad Sayeed to build off of their existing research about
  thematic fit. Samrat Halder and I continued this research extensively, and published a research paper at EMNLP BlackBox.
date: 9 Aug 2022
image:
  path: /assets/img/logo_emnlp.png
links:
  - title: Thesis
    url: https://arxiv.org/abs/2208.04749
  - title: Github
    url: https://github.com/MughilM/RW-Eng-v3-src
---
During my master's studies at Columbia University, it was required to spend a semester participating in a capstone
project along with other students, while under the mentorship of one or more mentors. A team of 5 students (including myself)
expressed interest to be placed under the mentorship of Yuval Marton and Professor Asad Sayeed.

## Final semester

The prior research of our mentors dealt with the subject of thematic fit. In a sentence, given the words and their "roles"
(akin to parts of speech), and a missing word with the role it's supposed to fill, how well do certain words fill that role?
For example, in the sentence `The child cut the cake with a <BLANK>`, words such as `knife` and `fork` work better
than `scissors` and `pliers`. Roles closely follow parts of speech, but terms differ and distinctions are made for
locations and other types of nouns. In this case, `child` is the **agent**, `cut` is the **verb**, the `cake` is the
**patient**, while the necessary role to fill is the `instrument`.

Our mentors built an existing language model that predicts the missing role and word simultaneously given the target
word and role, as well as the given word-role pairs present in the sentence. Embeddings were present for the input
words, roles, and the target words and roles. While the model was performing fairly adequately, the team of stundents
were tasked to find optimizations that will either make the model better or more efficient.

We focused on the embeddings throughout the model, as they were the source for the large number of parameters. 
Many optimizations were made, to improving the data ingestion pipeline, to combining embeddings for both the input
and target. Towards the end of the semester, a small presentation and poster session was held.

## Post-graduation
After graduation, another student (Samrat Halder) and I expressed interest in carrying this research further and 
producing a proper research paper. The further research extended the embedding optimizations by implementing
pre-trained embeddings by Meta (FastText) and Stanford (GloVe), and Google (Word2Vec). Evaluation on various
thematic fit tasks by controlling the initial embedding were researched and analyzed. 

Additionally, we also froze different parts of the network to pinpoint where the greatest learning occurred. 
Did it occur in the embedding, or did it occur in the few residual layers that came afterwards? Various experiments
were held using these settings.

All experiments were held on Google's Cloud Platform (GCP) and its virtual machines. 
We received research funding from Google for training purposes, as utilizing GPUs for accelerated training can be costly.
Big thanks to Google for providing all the resources. Finally, all experiments and analysis culminated in a research paper
that was submitted and accepted to the EMNLP BlackBox 2023 workshop. The paper is also available on arXiv. The abstract
has been reproduced below.


> Observing that for certain NLP tasks, such as
> semantic role prediction or thematic fit estimation, random embeddings perform as well as
> pretrained embeddings, we explore what settings allow for this and examine where most
> of the learning is encoded: the word embeddings, the semantic role embeddings, or “the
> network”. We find nuanced answers, depending on the task and its relation to the training objective. We examine these representation learning aspects in multi-task learning,
> where role prediction and role-filling are supervised tasks, while several thematic fit tasks
> are outside the models’ direct supervision. We
> observe a non-monotonous relation between
> some tasks’ quality score and the training data
> size. In order to better understand this observation, we analyze these results using easier,
> per-verb versions of these tasks.
> {:.lead}

