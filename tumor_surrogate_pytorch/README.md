# Geometry-aware neural solver for fast Bayesian calibration of brain tumor models (Pytorch implementation)


## Installation
- Python 3.6
- Install requirements: pip install -r requirements.txt
     
## Usage

To train a new model run:
    
    python3 -m tumor_surrogate_pytorch.train
    
To test the model and generate dice and mae histograms:

    python3 -m tumor_surrogate_pytorch.test
    
Training parameters can be adapted in tumor_surrogate_pytorch/config.py

## Citation

If you use our work, please cite https://arxiv.org/pdf/2009.04240.pdf
