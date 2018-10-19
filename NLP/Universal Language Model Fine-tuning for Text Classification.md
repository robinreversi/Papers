# Universal Language Model Fine-tuning for Text Classification (May 23, 2018)

Authors: Jeremy Howard, Sebastian Ruder



### Abstract

- Transfer learning for NLP tasks
- **with only 100 labeled examples, it matches the performance of training from scratch on 100× more data.**



### Introduction

- Fine tuning pre-trained word embeddings => good results
- NLP models are shallower than CV models
  - require different fine tuning methods
- **Propose new method: Universal Language Model Fine Tuning**
- **Key contributions:**
  - Discriminative fine-tuning
  - Slanted Triangular Learning Rates
  - Gradual Unfreezing
- Error reduction of 18-24% on the majority of datasets



### Universal Language Model Fine-Tuning

- Given static source task $\mathcal{T}_S$ improves performance on (different) target task $\mathcal{T}_T$ 
- **Argument: Language modeling is the most general source task, most akin to Imagenet**
- **ULMFiT advantages**
  - works across tasks varying in documents size, number, and label type
  - **single architecture and training process**
  - no custom feature engineering / preprocessing
  - does no require additional in-domain documents or labels
- **Use the SOTA language model AWD-LSTM** 
  - regular LSTM with no attention, short-cut connections
  - Authors believe can improve performance by using a more sophisticated model
- **ULMFiT Steps:**
  - General-Domain LM pretraining
    - **Pretrain on Wikitext-103**
    - 
  - Target task LM fine-tuning
  - Target task classifier fine-tuning
  - 

### 

### Results

- They do a lot better lol



### Analysis

- 



### Discussion and future directions

- Suggested Directions:
  - NLP for non-English languages
  - new NLP tasks where no SOTA exists
  - limited data
  - **Making pretraining more scalable**
    - Predicting far fewer classes
  - Multi-task
  - 