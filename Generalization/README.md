Here is a description of the files:

- `datasets_NODE.py` contains different functions to create 2D datasets.
- `models_NODE.py` contains classes corresponding to neural ODEs of order one and two, where the controls are MLP, and functions to train and evaluate the models.
- `models_NODE_piecewise.py` contains classes corresponding to neural ODEs of order one and two, where the controls are piecewise constant, and functions to train and evaluate the models.
- `script.py` contains a base code to train a model and visualize the results.

The code uses the library [torchdiffeq](https://github.com/rtqichen/torchdiffeq).
