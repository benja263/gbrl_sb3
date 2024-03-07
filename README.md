# # gbrl_sb3
GBRL based wrapper for stable_baselines3 algorithms
Gradient Boosting Reinforcement Learning

CPU & GPU compatible

## Getting started
Building cpu only docker
```
docker build -f Dockerfile.cpu -t <your-image-name:cpu-tag> .
```  
Running cpu only docker
```
docker run --runtime=nvidia -it <your-image-name:cpu-tag> /bin/bash
```  
Building gpu docker
```
docker build -f Dockerfile.gpu -t <your-image-name:gpu-tag> .
```  
Running gpu docker
```
docker run --runtime=nvidia -it <your-image-name:gpu-tag> /bin/bash
```  
## Features
### Tree Fitting
- Greedy (Depth-wise) tree building - (CPU/GPU)  
- Oblivious (Symmetric) tree building - (CPU/GPU)  
- L2 split score - (CPU/GPU)  
- Cosine split score - (CPU/GPU) 
- Uniform based candidate generation - (CPU/GPU)
- Quantile based candidate generation - (CPU/GPU)
- Supervised learning fitting / Multi-iteration fitting - (CPU/GPU)
    - MultiRMSE loss (only)
- Categorical feature support
### GBT Inference
- SGD optimizer - (CPU/GPU)
- ADAM optimizer - (CPU only)
- Control Variates (gradient variance reduction technique) - (CPU only)
- Shared Tree for policy and value function - (CPU/GPU)
- Linear and constant learning rate scheduler - (CPU/GPU only constant)
- Support for up to two different optimizers (e.g, policy/value) - **(CPU/GPU if both are SGD)




