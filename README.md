These notebooks contain experiments that are described in my paper submitted to ECML 2017 conference.

# Dimensionality Reduction and Visualization Using Evolutionary Algorithms and Neural Networks

In this paper, we propose a novel method for a supervised dimensionality reduction, which learns weights of a neural network using an evolutionary algorithm, CMA-ES, in combination with the k-NN classifier. If no activation functions are used in the neural network, the algorithm essentially performs a linear transformation. This linear neural network can also be used inside of the Mahalanobis distance and therefore our method can be considered to be a metric learning algorithm. By adding activations, the algorithm can learn non-linear transformations as well. We consider reductions to low dimensional spaces, which are useful for data visualization, and demonstrate that the resulting projections provide better performance than other dimensionality reduction techniques and also that the visualizations provide better distinctions between the classes in the data thanks to the locality of the k-NN classifier. 

# Structure

The experiments are split into several jupyter notebooks:
- [01]: the datasets used in the paper can be downloaded here, but they are also available in the *datasets* folder
- [02]: improving **success rate of kNN classification experiment**
- [03]: extracting results and plotting graphs from the [02] experiment
- [04]: **dimensionality reduction experiment**
- [05]: generalization of the evolution experiment [not in the paper]
- [06]: measuring learning times on datasets from [01] experiment [not in the paper]
- [07]: measuring run times on continuously increasing dimensions experiment [not in the paper]

The raw results from the experiments are saved in *results* folder. The resulting graphs are in the *graphs* folder.
