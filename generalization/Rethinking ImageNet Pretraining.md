# Rethinking ImageNet Pre-training

Authors: Kaiming He Ross Girshick Piotr Dollar 

https://arxiv.org/pdf/1811.08883.pdf

**Key Takeaway:** Performance not increased by pretraining on ImageNet when tasks are dissimilar and enough data (surprisingly less than previously thought) is availaible. 



### Abstract

- Increasing number of iterations allows models trained from scratch to reach performance levels of pretrained models
- Holds for up to 10% of COCO dataset



### Introduction

- Benefits of larger datasets currently unclear 
- In this paper,
  - Showed large models can be trained from scratch without overfitting
  - Showed getting to same performance level takes about as much time as ImageNet pre-training plus fine-tuning
  - ImageNet doesn't give better regularization when data is sufficient
  - ImageNet pretraining doesn't help as much with spatially sensitive / "well-localized" predictions



### Related Work

- Existing literature supports ideas that improvements on obj det are small from pretraining on ImageNet  / comparable databses 
- Other work has tried to develop models that specialize in training from scratch but did not investigate whether such specialization was necessary



### Methodology 

- To allow models to train successfully from scratch, need to replace batch norm (BN) with group norm (GN) or synchronized batch norm (SyncBN)
- Also showed that with appropriate normalized initialization, can train object detectors with **VGG** without BN or GN
  - **BN / GN only important for very deep residual networks?**

- Also found that to allow models to train from scratch, longer training periods are necessary to allow the model to see roughly the same number of examples, arguably in terms of pixels



### Experimental Settings

- Baselines and hyper-parameres follow Mask R-CNN paper
- Architecture:
  - Mask R-CNN with ResNet or ResNeXT + FPN
  - **using retrained GN outperformed frozen BN layers**
- Learning Rate Scheduling:
  - reduce learning rate by 10x in the last 60k and 20k respectively 
  - training for longer on the first (large) learning rate useful
  - **training for longer on small learning rates leads to overfitting**



### Results

- See paper for details, but essentially training from scratch matches performance of models pretrained on ImageNet on a wide array of baselines / metrics / augmentations
- Suggests **"using good optimization strategies and training for longer are sufficient for training comparably performant detectors on COCO"**
- Training from scratch with less data
  - using 35k or 10k images, still achieves similiar performance from scratch 
  - using 1k images, worse generalization but same optimization speed (in terms of training loss)
- Doesn't achieve same results on PASCAL VOC, but authors argue images have less instances per image and less categories, so in total less data



