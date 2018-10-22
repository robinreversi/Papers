# Speeding up the Hyperparameter Optimization of Deep Convolutional Neural Networks

Authos: Tobias Hinz, Nicolas Navarro-Guerrero, Sven Magg and Stefan Wermter

https://arxiv.org/pdf/1807.07362.pdf

**Key Takeaway:** Good hyper parameters for Deep CNNs can be approximated / found more quickly by using lower scale images independent of the optimization method.



### Abstract

- Use a lower-dimensional representation of the data to search for HP
- Test over
  - Random Search
  - Tree of Parzen Estimator's (TPE)
  - Sequential Model-Based Algorithm Configuration (SMAC)
  - Genetic Algorithms (GAs)



### Introduction

- High resolution images good for performance, but smaller low resolution images good for speed
  - **Solution:** hyperparameter search using low resolution images, fine-tune with high resolution images
  - **Connection:** similar to idea that hyperparameters for one dataset might be good for another
    - **Question:** implies hyperparameters are somewhat tied to the model? What's the connection here?
- Key Experiments
  1. Dependencies between hyperparameters found on different input sizes
     - **Same hyperparameters important regardless of image resolution**
     - **Optimal values seems to be independent regardless of image resolution**
  2. Can we use this information to speed up HP search?
     - Start from small images and scale up



### Related Work

- Most widely used methods:
  - Grid Search
    - Bad, exponential in growth with number of HPs
  - Random Search
    - Random value for each HP
    - with large number of HPs, much faster than Grid Search since less time potentially spent on unimportant HPs
  - **Hyperband**
    - extension of random search
    - samples a set of HP configuration
    - configurations then trained for a number of iterations and then ranked on performance
    - take best $k$ configurations
      - repeat until only few remaining, then training all
- Sequential Model-Based Observation (SMBO) 
  - Approximate next best HP setting to save time training
  - Build a probabilistic model describing the performance of the learning algorithm $f$ given a HP setting
    - Take advantage of any priors
    - Continuous update with point values of $f$
- Evolutionary / swarm algorithms
  - Run many candidates in parallel
  - "weaker" / poorer performance dies off / replaced by some subset of the better settings
- Reinforcement Learning
  - Usually oriented around model architecture, not learning rate / regularization parameters

### Methodology 

- All optimization algorithms evaluated with 1500 hyperparameter settings per optimization run
- Evaluate on the original images and on the rescaled images
  - Increasing Image Sizes (IIS) for the latter
- Each approach repeated three times for each dataset and optimization procedure
  - Optimized the following hyperparameters:
    - Learning Rate
    - \# Conv / FC Layers
    - \# Filters per conv layer and size
    - \# Units per FC layers
    - Batch Size
    - L1 and L2 regularization parameters
  - Stop training process if performance does not improve on val set for 5 epochs
- CNN architecture layout
  - Each convolutional layer followed by max-pooling layer which reduces the input size by a factor of four
  - After last conv layer, FC layer with softmax activation
- **First Experiment -- Does the importance of hyperparameters stay constant across different optimization algorithms?**
  - Need to be able to scale down images, so can't use MNIST or CIFAR
- **Second Experiment -- Does using IIS increase model training speed without decreasing model performance?**
- Datasets:
  - Cohn-Kanade (CK+) -- facial expressions of 210 adults
    - converted to gray scale
    - resized to 200x200, 128x128, 64x64, 32x32
    - Splits:
      - 70% Train, 30% Val **by class**
  - STL-10 -- labeled images acquired from ImageNet
    - Color Images 96x96, 10 classes, 500 examples for training in each class, 800 images for testing purposes
    - Converted the images to grayscale, rescale to sizes 32x32, 48x48
    - Pre-defined folds with 100 images from each class in train (total 1000 images)
    - **Test accuracy calculated as the average of each of 10 models on the test set**
      - based on each fold??
    - **Use first fold as training data and remaining in train set repurposed to val set**
  - 102 Flowers 
    - High resolution (min 500x500) => 128x128 as final input size (model should perform well with these inputs)
    - Pre-defined split => 2040 for training / val |  6149 for test
    - First 1020 in train used to optimize HP | other 1020 used to evaluate performance

### Importance of Hyperparameters

- (random search, TPE, GA) all found similar values for HP for all resolutions

  - Interesting, suggests there is some kind of a optima for HPs? 
  - LR, batch size, L1 / L2 regularization weights, number of layers very consistent
  - Minor differences in number of filters, units per filter, units per convolution, FC layer
  - **Hyperparameters found by TPE and GA perform better**

- **Evaluation Criteria:**

  - functional ANOVA 
    - Analysis of variance
    - **The importance of a set of hyperparameters is defined as the amount of variance it accounts for**
      - How do they find the most important subsets?
    - Fig 2:

  ![](/Users/robincheong/Documents/Stanford/Papers/To Read/Speeding up the Hyperparameter Optimization of Deep Convolutional Neural Networks/fig2.png)

- Most important hyper-parameters stay largely the same, but seems like the importance of the number of hidden layes / conv layers 
  - Could be since those depend directly on the input dimension
  - Could also be because more layers => more parameters meaning more potential to overfit?
  - Suggests optimal number of layers needs to be fine tuned to the data size
    - Authors suggest to infer the number of filters per convolutional layer based on # filters for smaller input sizes
- The impact of changing the learning rate is fairly consistent across the various input dimension sizes
- Even the actual "best" value for the LR and # hidden layers is fairly consistent across input dimension sizes
- On a held out test set, the GA and TPE algorithms outperformed random search with no additional regularization methods applied
  - Test results similar to validation results which implies hyperparameters generalize well, not overly fit to the validation set

### Using IIS for Hyperparameter Optimization

- The process:
  - 32x32 for 750 iterations
  - Hyperparameters from 32x32 used to optimize 64x64 for 500 iterations
  - Hyperparameters from 64x64 used to optimize 128x128 for 250 iterations
  - **In theory faster than optimizing 128x128 for 1500 iterations**
- Between 9% to 42% time reduction
- For Random Search, GA, IIS => lower error rates and faster convergence
  - Possibly because after every rescale, narrow down HP space to good value ranges
- On a held out test set, all settings perform about the same / well
  - suggests IIS independent of underlying optimization algorithm

### Conclusion

- Main Advantages:
  - easy to use
  - easy to parallelize and combine with other HP optimization techniques (Fabolas and Hyperband?)
- Work to be done:
  - **Does it work on non-image datasets / non-CNN models?**