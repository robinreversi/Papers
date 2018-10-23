# On the Importance of Single Directions for Generalization (May 2018)

Authors: Ari S. Morcos, David G.T. Barrett, Neil C. Rabinowitz, & Matthew Botvinick

https://arxiv.org/pdf/1803.06959.pdf

**Key Takeaway:** To generalize well, a model shouldn't depend on the outputs of a single unit. In addition, preliminary findings show that highly specific filters tuned to particular classes might result in poorer generalization.

### Abstract

- What's the difference between networks that generalize well and ones that don't?
- This paper argues that a network's dependence on the responses of single neurons (single directions) is a good predictor of generalization performance
  - Network's that are highly dependent on a single neuron's response generalize poorly
- Dropout less effective than batch norm at decreasing dependencing on single directions
- Individually selective units may not be necessary for strong performance



### Introduction

- Some work finds flatness of minima makes an impact but some say sharp minima also generalize well
- **Single direction: activation of a single unit or feature map in response to some input**
  - Depending on single directions => overfitting / poor generalization
    - Can potentially be used to determine when to stop training
- Batch norm helps by decreasing class selectivity of individual feature maps



### Approach

##### Models and Datasets

- 2-Layer MLP on MNIST
- 11-Layer CNN on CIFAR-10
- 50-Layer ResNet on ImageNet
- ReLU NonLinearities
- BatchNorm for CNNs

##### Perturbation Analyses

- Ablations:
  - Measure importance of a single direction by measure change in performance once direction is removed
    - Remove by clamping values to a constant **post activation**
  - **Found that clamping value to mean activation more damaging than clamping to zero**
- Addition of Noise:
  - Test reliance upon random single directions instead of coordinate-aligned single directions
    - Add random noise with increasing variance to all units

##### Quantifying Class Selectivity

- Calculate the <u>class conditional mean activity:</u>

  ​		$ selectivity = \frac{\mu_{max} - \mu_{-max}}{\mu_{max} + \mu_{-max}}$

  where $\mu_{max}$ is defined as the highest class-conditional mean activity and $\mu_{-max}$ is the mean activity across all other classes
  - Varies from 0 to 1 with 0 meaning a unit's average activity is the same for all classes and 1 meaning a unit is only active for inputs of a single class
  - Authors note this is an imperfect metric since a unit with a little information about every class would have low selectivity 
    - **Question:** isn't that intended?
      - Authors show that mutual information which highlights units with multiple classes produces similar outcomes

### Experiments

##### Generalization 

- Intuition:

  - Consider two networks, one that memorizes the data and one that learns the structure inherent in the data
  - To fully describe the network that just memorized the data requires more information than to describe the one that found inherent patterns
  - Implies memorizing requires using more of a network's capacity and (**AUTHOR'S ARGUMENT**) thus single directions
  - Thus single direction pertubations affect memorizing networks more than pattern finding ones

- To test this theory, trained networks on data sets where varying % of labels were replaced with noise and plotted train accuracy versus (number units ablated / per-unit noise)

  - Showed memorizing networks much more sensitive to ablations / noise (networks that have noisier labels)
  - Not perfect correspondence 

- What about on non-corrupted datasets, does the same pattern hold true?

  - To test, trained 200 networks with different initializations, data order, and learning rates
  - Then took top / worst 5 and performed ablations / random noise
  - **Found:** best generalizing networks also most resistant to ablations and random noise 


##### Reliance on Single Directions as a Signal for Model Selection

- **Can single direction reliance be used to estimate generalization performance?**
- Found same point at which AUC of cumulative ablation curve decreased same point at which test loss began increasing
- Found negative correlation between AUC  and test loss 
- Testing whether single directions can be used to estimate good hyperparameters found that AUC and test accuracy high correlated

##### Relationship to Dropout and Batch Normalization

- Dropout
  - Seems like a good idea at first for reducing single direction dependence since removing directions is akin to dropping those parameters out 
  - but...
    - Memorizing network can distribute a single direction to several other directions
      - **Question:** Doesn't that mean it's no longer depending as much on a single direction?
    - Network only encourage to make minimum number of copies of single directions
  - To test, trained MLPs on MNIST with dropout on both corrupted and uncorrupted labels
    - Networks with dropout required more epochs to converge and converged to worse solutions at higher probabilities => dropout discourages memorization
    - Up to dropout percentage, minimal loss in training accuracy but once past that point, networks trained with noisy labels much more sensitive to ablations than uncorrupted labels
    - Suggests dropout only helps to an extent determined by the fraction of nodes dropped during training
- Batch Norm
  - Empirically seems to discourage over reliance on single directions
  - Networks trained with BN consistently more robust to ablations 

##### Relationship between Class Selectivity and Importance

- Is the class selectivity of a unit related to its importance to the network's output?
- First: does BN affect class selectivity / affect distribution of information
  - Class selectivity of units in networks trained with BN substantially lower
  - BN seems to increase mutual information in feature maps
  - **Raises question about whether highly selective feature maps are beneficial**
- Ablation of highly selective units has little effect on model performance
  - Found that early layers had negative correlation between ablation and model performance (i.e removing these selective units improved performance)
  - Later layers had no relationship
  - Found same results with mutual information

- compared selectivity to $L^1$ norm of the filter weights which is an indicator of the importance of the feature map
  - Found no relationship and if anything a negative one

### Related Work

- This work supports the idea that flat minima generalize well since pertubations along single direction lead to little effect in networks that generalize
- Information theory perspective argues memorizing networks store more info in their weights
  - This paper supporsts that argument since less compressed networks should be more reliant upon single directions

### Discussion and Future Work

- Construct a regularizer that more directly penalizes reliance on single directions
  - Most obviously dropout but showed this doesn't work well
- Could use this to predict generalization performance without a separate val / test set
  - Possible to use to gauge when to perform early stopping
  - Remove need for as many labeled data points
- Another direction: what about test sets that come from overlapping distributions as the train set? do models that generalize well still depend on single directions?
- Class selectivity of single directions is not well correlated with their importance to network's output
  - Analyzing single inputs / units is not a good way to understand model behavior
- Batch norm decreases class selectivity 