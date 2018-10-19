# Universal Language Model Fine-tuning for Text Classification (May 23, 2018)

Authors: Jeremy Howard, Sebastian Ruder

https://arxiv.org/pdf/1801.06146.pdf

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
      - 28,595 preprocessed Wikipedia articles
      - 103 million words

  - Target task LM fine-tuning

    - Discriminative fine-tuning

      - **Argument: Different layers capture different types of information, so they should be fine-tuned to different extents**

      - **Tune each layer with different learning rates**

      - Split parameters for each layer into $\{ \theta^1, \dots, \theta^L\}$ and train using $\{ \eta^1, \dots, \eta^L\} $

      - Empirically found choosing the learning rate of the last layer first then setting 

        $\eta^{l-1} = \eta^l / 2.6$ to work well. 

    - Slanted triangular learning rates

      - Initially, losses are going to be high and noisy so don't start the learning rate high
      - Instead, increase learning rate linearly from zero to desired starting learning rate so that initial losses don't wipe out pretraining
        - **Argument** also helps with allowing the model to quickly adjust to the new input space 
      - Akin to triangular learning rates (Smith 2017) except shorter increase period and longer decrease period
      - See paper for implementation details

  - Target task classifier fine-tuning

    - Augment the pretrained model with two linear blocks
      - Each linear block uses batch norm, dropout, ReLU activation
      - Softmax activation outputs a probability distribution over target classes 
      - These layers trained from scratch
      - **First linear layer takes as input the pooled last hidden layer states**
    - Concat pooling
      - **Argument** signal contained in only a few words, occur anywhere in the document
      - To prevent loss of information, don't just use the last hidden state
      - Instead, concatenate the last hidden state $h_T$ with the max-pooled and mean-pooled representation of the hidden states over as many time steps as fit in GPU memory $H = \{h_1, \dots, h_T\}:
        - $h_c = [h_T, maxpool(H), meanpool(H)]$
        - Needs clarification
      - Finetuning is critical
        - too fast => wipe out pretrained weights
        - too slow => slow convergence, overfitting
    - Gradual unfreezing
      - Rather than fine-tuning all layers, freeze everything except last layer (which contains the least general knowledge) 
      - Train for one epoch
      - Unfreeze next layer
      - Repeat for all layers, then until convergence
      - Similar to "chain-thaw" (Felbo 2017)
    - **Argument: ** from empirical observations, these techniques complement each other
    - BPTT for Text Classification (BPT3C)
      - Normally trained with backpropagation through time (BPTT) for large input sequences
      - To fine-tune a classifier for large documents, authors propose BPT3C
        - divide document into fixed length batches of $b$
        - at the beginning of each batch, model is initialized with final state of prev batch, keeping track of hidden states
        - gradients back propagated to batches whose hidden states contributed to the final prediction
        - In practice, **use variable length backpropagation sequences** (Merity 2017a)
    - Train both a forward and backward LM and fine-tune classifier for each and average predictions

### Experiments



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