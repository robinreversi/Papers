# Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour (Apr 2018)

Authors: Priya Goyal; Piotr Dollar; Ross Girshick; Pieter Noordhuis; Lukasz Wesolowski; Aapo Kyrola; Andrew Tulloch; Yangqing Jia; Kaiming He 

https://arxiv.org/pdf/1706.02677.pdf

**Key Takeaway:** It's possible to train on large minibatches. Problems before were within optimization not within generalizatio

### Abstract

- Developed a way to synchrnously train on large distributed batches of data 
- Shows no decrease in performance even with batch sizes of 8192
- Trains on ImageNet in 1 Hour



### Introduction

- Train on large batches and dramatically reduce training time
- Employ hyper-parameter free linear scaling rule
- New warmup strategy
- Experiments show optimization (not generalization) is the issue with large minibatches 

### Large Minibatch SGD

- Advantages: scale to multiple workers without reducing per-worker workload or model accuracy
- **Linear Scaling Rule:** when the minibatch size is multiplied by k, multiply the learning rate by k
  - Helps not only match accuracy but also training curves allowing for easy debugging / comparison
  - **Intuition on why this works:** Imagine small batch SGD takes many small steps in a single direction. To compensate, large batch SGD then needs to take a much larger step, namely the ratio of the size of the large batch to the small. 
    - **KEY ASSUMPTION:** The overall direction of the gradient updates between several small batches and one big batch is relatively the same
    - See paper for equations and details
    - **ASSUMPTION FAILS WHEN NETWORK IS INITIALLY TRAINING AND FOR EXTREMELY LARGE MINIBATCHES (>8K)**
- **Warm-up**
  - Constant Warmup
    - Low learning rate for first few epochs
    - Good for models with a mix of pretraiend and new layers
    - **Not sufficient when multiplier k is large** (learning rate jumps suddenly)
  - Gradual Warmup
    - Gradually increase learning rate from $\eta$ to $k\eta$ . 
- **Batch Normalization with Large Minibatches**
  - BN breaks independence of each sample's loss whereas SGD assumes each loss is independent
  - Instead, consider X as being subdivided into subsets of size n where the loss for each subset is computed independently of the others
  - Changing the batch size then fundamentally changes the loss function being optimized
  - Important then to keep the number of samples per worker constant even when increasing the batch size
    - i.e increase number of workers rather than number of samples per worker
  - In practice used n=32 

### Subtleties and Pitfalls of Distributed SGD

- **Weight Decay**

  - Scaling the cross-entropy loss is not the same as scaling the learning rate
    - ex: when there's weight decay

- **Momentum Correction**

  - Two different forms:

  - $u_{t+1} = mu_t + \frac{1}{n} \sum_{x\in\mathcal{B}} \nabla l(x, w_t)$

    $w_{t+1} = w_t - \eta u_{t+1}$ 

  - $v_{t+1} = mv_t + \eta \frac{1}{n} \sum_{x\in\mathcal{B}} \nabla l(x, w_t)$

    $w_{t+1} = w_t - v_{t+1}$

  - 2nd incorporates learning rate into hidden state

  - Equivalent for fixed $\eta$ but when $\eta$ grows quickly, history $v_{t}$ gets dwarfed by new update.

  - **Apply Correction:**  $v_{t+1} = m \frac{\eta_{t+1}}{\eta_t} v_t + \eta_{t+1} \frac{1}{n} \sum_{x\in\mathcal{B}} \nabla l(x, w_t)$

- **Gradient aggregation**

  - Need to aggregate gradient updates from all workers over all examples of each worker
  - Each worker computes loss over their own batch
  - Need to also average then over workers
  - Communication primitives perform adding though not averaging
  - So move the $\frac{1}{k}$ factor into the loss i.e **normalize per worker loss by total minibatch size loss kn instead of n** 

- **Data Shuffling**

  - SGD can sample with or without replacement
  - Doing one shuffle and dividing the training data after is better

### Communication

- Largely systems

### Main Results and Analysis

- **MAIN RESULT:** train ResNet-50 on ImageNet in 1 hour matching accuracy of small minibatch training

- **Experimental Settings**
  - Dataset: ImageNet
  - ResNet-50 with stride-2 convs on 3x3 layers instead of 1x1
  - Nesterov Momentum (but standard momentum equally effective)
  - Weight decay: $\lambda = .0001$
  - n = 32 (num examples per worker)
  - All trained for 90 epochs
  - Linear Scaling Rule with $\eta = .1 * \frac{kn}{256}$ (reference learning rate)
    - reduce by .1 at 30th, 60th, 80th epoch
  - Initialize conv layers based on "Delving deep into rectifiers: Surpassing human-level performance on imagenet classification", FC layer based on zero-mean Gaussian with std .01
    - Initialization affects large minibatches moreso than small (?)
  - BN Layers, $\gamma = 1$ except for in the residual block's last BN layer which is initalized to 0
    - causes forward / backward signal to propogate through identity shortcuts
  - Data augmentation: 
    - 224x224 random crop from augmented image or flip
    - normalize by per-color mean and standard deviation
  - **Random Variation**
    - compute model's error as median of final 5 epochs and report mean and std of 5 independent runs
- **Optimization or Generalization Issues**
  - Experiments show large minibatch matches training and validation of small minibatch
  - Also show that transfers well to other tasks like obj det and segmentation
  - k = 256, n = 32, kn = 8k
  - Baseline has kn=256 and $\eta = .1$
    - With linear scaling rule $\eta = 3.2$
  - Test three warm up strategies:
    - none | constant $\eta = .1$ for 5 epochs | gradual warmup $\eta=.1$ and increased linearly to $3.2$ over 5 epochs
    - Avoids hyperparameter tuning to show that you can increase speed of training by increasing minibatch size without sacrificing accuracy or needing more hyperparameter tuning 
    - **Training Error**
      - gradual warm up necessary to have large minibatch sizes match small minibatch
      - constant warm up degrades results
    - **Validation Error**
      - No generalization degradation if using gradual warm up
- **Analysis Experiments**
  - **Minibatch vs Error**
    - Minibatch size does not affect error from 64 to 8k, increases after
    - When large minibatch training curve diverges from small, validation performance also suffers
  - **Alternative Learning Rate Rules**
    - Small minibatches (<256) $\eta=.1$ is best
    - Linear scaling rule works (8k, $\eta = .1 * 32$ gives optimum error as well) 
    - Changing learning rates changes shape of curve but not optima
  - **Batch Norm Initialization**
    - Marginally increases performance
    - more important for large mini batches
  - **Resnet101**
    - about the same as ResNet50 (maxes out at 8k)
  - **ImageNet 5k**
    - larger, is the limit of the batch size determined by information content?
      - no, increasing data does not allow you to increase batch size
  - **Generalization to Detection and Segmentation**
    - Yes, so long as you don't lose generalization performance on the classification
- **Runtime**
  - Linear decrease in time with higher batch size but smaller batch sizes faster up to 2k

