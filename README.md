# Context-Dependent Gating for RNNs

This code accompanies our paper:
> Alleviating catastrophic forgetting using context-dependent gating and synaptic stabilization  
> Nicolas Y. Masse, Gregory D. Grant, David J. Freedman  
> https://arxiv.org/abs/1802.01569

The feed-forward network model can be found in the repository:
> https://github.com/nmasse/Context-Dependent-Gating/

Dependencies:  
> Python 3  
> TensorFlow 1+  

In our paper, the model is tested on a set of cognitive tasks proposed by Yang et al.
> https://www.biorxiv.org/content/early/2017/09/01/183632
>
> Our tasks are set up in the stimulus.py file, and are cycled through automatically with the current configuration.  Note that our model uses one-hot outputs as opposed to distributions, as originally described by Yang et al.
