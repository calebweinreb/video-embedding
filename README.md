# Video clip embedding

The `train.ipynb` notebook contains code for training a 3D convnet using Barlow Twins. The goal is to embed short video clips of animal behavior in a semantically-meaningful latent space. 

## Development environment

Create a virtual environment using conda:
```
conda create -n video_embedding python=3.10
conda activate video_embedding
```
Install `ipykernel` to use the conda environment in Jupyter notebooks:
```
pip install ipykernel
python -m ipykernel install --user --name=video_embedding
```
When you want to use the conda environment in Jupyter notebooks, select the `video_embedding` kernel.


## TODO:

1. Install necessary dependencies to run the full notebook. List all the dependencies you had to install in a file called `requirements.txt`

2. Add comments/docstrings that explain the purpose of each function (or question marks if any you're not sure). 

3. Create a minimal python package by adding an `__init__.py` file and splitting the functions into separate files that cover different functionalities (e.g. `io.py` for video reading and writing, `augmentation.py` for augmentations, `model.py` for the model, etc.). The package should be importable from the notebook.

4. Add formal docstrings to all the functions and classes. Use the Google style guide for docstrings. Refactor where necessary to make the code more readable.

5. Formalize the python package by adding a `setup.py` file and versioning (Caleb will do this).

6. Add functionality for animal tracking / cropping. 

7. Create documentation, including installation instructions and a tutorial notebook. 