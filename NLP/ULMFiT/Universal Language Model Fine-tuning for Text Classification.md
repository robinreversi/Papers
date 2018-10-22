# Universal Language Model Fine-tuning for Text Classification (May 23, 2018)

Authors: Jeremy Howard, Sebastian Ruder

https://arxiv.org/pdf/1801.06146.pdf

**Key Takeaway:** Transfer learning for NLP is possible, shows great results, and is more about how to perform the optimization process than anything inherent within the model.

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

#### Set up

- **Focus:** text classification tasks
- 6 Datasets used for sentiment analysis, question classification, topic classification
- **Sentiment Analysis**
  - Binary movie review IMDb dataset
  - Binary Yelp Review Dataset
  - 5-Class Yelp Review Dataset
- **Question Classification**
  - 6-Class TREC dataset
- **Topic Classification**
  - AG News
  - DBPedia ontology datasets
- **Preprocessing** as in Johnson and Zhang 2017, McCann 2017
  - also add special tokens for capitalization, elongation, repetition
- **Hyperparameter Tuning**
  - Same set of HP across tasks unless otherwise mentioned
    - **Motivation**: 	Model should perform well across a variety of texts ("general knowledge")
  - AWD-LSTM language model
    - Embedding Size: 400
    - 3 Layers
    - 1150 hidden activations per layer
    - BPTT size of 70
    - Dropout 
      - .4 to layers
      - .3 to RNN layers
      - .05 to embedding layers
      - Weight dropout of .5 to RNN hidden-to-hidden matrix
    - Classifier has hidden layer of size 50
  - Adam Optimizer
    - $\beta_1 = .7$
  - Batch Size 64
  - Fine Tuning Base Learning Rate .004 for LM, .01 for classifier
  - Tune number of epochs based on validation set
  - Otherwise same as Merity 2017a
- **Compare against current SOTA**
  - IMDb and TREC-6, compare against CoVe (another transfer learning technique)
  - AG, Yelp, DBPedia compare against Johnson and Zhang (2017)

#### Results

- All measures reported as error rates
  - **Limitation:** obscures performance on different classes?
    - Confusion matrix might've been helpful?
- Outperforms CoVe and other SOTA models on 
  - IMDb (reduced error by 43.9% compared to CoVe and 22% compared to SOTA)
    - CoVe (8.2) -> Virtual (5.9) -> **ULMFiT** (4.6)
  - TREC-6
    - Can't claim statistical significance b/c test set is small but performs about the same
    - Outperforms McCann's (pretraining?) approach despite using two orders of magnitude less data
    - CoVe (4.2) -> LSTM-CNN (3.9) -> **ULMFiT** (3.6)
  - AG
    - DPCNN (6.87) -> **ULMFiT** (5.01)
  - DBpedia
    - DPCNN (.88) -> **ULMFiT** (.80)
  - Yelp-bi
    - DPCNN (2.64) -> **ULMFiT** (2.16)
  - Yelp-full
    - DPCNN (30.58) -> **ULMFiT** (29.98)
  - **Significance:** A pre-trained LSTM with dropout outperformed models with attention, sophisticated embeddings, complex architectures
  - **Question:** Why are there notable performance gains on some datasets / tasks but not others? Related to similarity of the tasks?
- **Claim:** IMDb is reflective of real world data
  - Varying document lengths / types

### Analysis

- Ablation experiments and other analyses
- Three datasets: IMDb, TREC-6, AG
- **Only train / validation set?**
  - 10% of train used to report test measurements using a unidirectional LM
- Fine-tune classifier for 50 epochs and train all methods **except ULMFiT** with early stopping
- **Low-shot learning**
  - Tested two modes:
    - Supervised -- only labeled examples used to fine tune
    - Semi-supervised -- all data (including unlabeled) used to fine tune
    - Training from scratch
  - used same hyperparameters as before
  - **"On IMDb and AG, supervised ULMFiT with only 100 labeled examples matches the performance of training from scratch with 10× and 20× more data respectively"**
  - **"If we allow ULMFiT to also utilize unlabeled examples (50k for IMDb, 100k for AG), at 100 labeled examples, we match the performance of training from scratch with 50× and 100× more data on AG and IMDb respectively."**
  - ![](/Users/robincheong/Documents/Stanford/Papers/NLP/ULMFiT/ULMFiT_1.png)

- **Impact of Pretraining**
  - No pretraining vs. pretraining on WikiText-103
  - Pretraining improves performance even on large datasets
  - **Sometimes substantially (10.67->5.69 on TREC-6), sometimes not (5.52->5.38 AG)**
  - More research necessary
- **Impact of LM Quality**
  - Tested vanilla LSTM vs AWD-LSTM LMs
  - Showed model architecture improves performance
- **Impact of LM fine-tuning**
  - no fine-tuning vs fine-tuning the full model
  - with / without discriminative fine-tuning
  - with / without slanted triangular LR
  - **Most effective on large datasets**
  - improves performance generally
- **Impact of classifier fine tuning**
  - Scratch v fine-tuning full model v only last layer v chain thaw v gradual unfreezing
  - also compare discriminative LRs and triangular learning rates with aggressive cosine annealing scheduler (Loshchilov and Hutter, 2017).
  - $\eta^L = .01$ for discriminative LR, .001 and .0001 for last and all other layers for chain-thaw, and .001 otherwise
    - **Limitation:** what if other learning rates worked better for these models?
  - Fine-tuning >> scratch
  - Fine-tuning only last layer underfits
  - gradual unfreezing, chain-thaw, and full fine-tuning all perform about the same
  - Disriminative LRs improves performance of both full fine-tuning and gradual unfreezing except on AG
  - Cosine Annealing comparable with slanted triangular LRs, but underperforms on smaller datasets
  - ULMFiT (with slanted triangular LRs, gradual unfreezing, discriminative LRs) performs well across the board
    - **Argument:** implies universally good
      - But not a fair comparison -- very hand tailored, possible that another combination might work well across the board too
- **Classifier fine-tuning behavior**
  - fine-tuned with ULMFiT versus fine-tuning full model
  - **fine-tuning full model causes it to converge faster but suffer from "catastrophic forgetting"**
  - **ULMFiT comparable stable**
- **Bidirectionality**
  - it helps (a bit: .5-.7 improvement)

### Discussion and future directions

- Suggested Directions:
  - NLP for non-English languages
  - new NLP tasks where no SOTA exists
  - limited data
  - **Making pretraining more scalable**
    - Predicting far fewer classes
  - Multi-task
  - 