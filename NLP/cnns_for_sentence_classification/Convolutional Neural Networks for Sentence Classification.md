# Convolutional Neural Networks for Sentence Classification (EMNLP Sep 2014)

Authors: Yoon Kim

https://arxiv.org/abs/1408.5882

**Key Takeaway**



### Abstract

- Using CNNs on pre-trained word vectors for sentence classification
- Static vectors, little HP tuning
- Fine-tuned task-specific vectors improves performance



### Introduction

- One layer CNN on top of word vectors from an unsupervised neural language model
- **Argues: **word vectors are 'universal' features akin to features early on in deep CNNs



### Model

Let $x_i \in \R^k$ represent the ith word in the vocabulary where $k$ is the size of the feature vector. We can define  a sentence then as the concatenation of all of the feature vectors of its words. 

We can then convolve a filter $w \in \R^{hk}$ over the sentence then where $h$ is the number of words we look at a time to output a feature $c$. We can use a stride of one to convolve over every possible window of words and output a total feature map $c = [c_1, c_2, \dots, c_{n-h+1}]$. 

In this work, the authors take the maximum value feature via a max-over-time pooling operation to select a particular feature to correspond to this filter. They argue this captures the most important feature for each feature map and allows the model to deal with variable sequence lengths (I'm assuming this means padded values are hopefully discarded?)

Use multiple filters of various window sizes to get several feature maps. Feed these into a FCN to get outputs.

Authors attempt to incorporate both static and fine-tuned word vectors by treating them as separate channels. 

Authors use dropout and weight clipping ($||w|| > s$, rescale w to have $||w|| = s$) as regularization.



### Datasets and Experiment Setup

- Datasets
  - Movie Reviews 
  - SST-1 / SST-2 - Stanford Sentiment Treebank
    - Similar to Movie Reviews but with better labels
  - Subj - Subjective or Objective
  - TREC Question Dataset
    - Classify question into 6 question types
  - CR
    - Predict customer reviews
  - MPQA 
    - Polarity detection subtask

- Hyperparameters and Training
  - ReLUs, filter windows of 3, 4, 5 with 100 feature maps each, dropout rate .5, $l_2$ constraint of 3, mini-batch size 50
  - Early stopping, SGD with Adadelta
- Word2Vec 
  - Words not present initialized randomly
- **Model Variations**
  - CNN-rand
    - Words initialized randomly and then trained
  - CNN-static
    - Words pre-trained and fixed
  - CNN-non-static
    - Words pre-trained and fine-tuned
  - CNN-multichannel
    - Two copies of the words, one pre-trained and the other fine-tuned



### Results

- Pretraining necessary
- Multichannel inconclusive
- Fine-tuning allows for more meaningful representations
- Par performance with many SOTA models at the time
- 