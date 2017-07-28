# Multiagent DQN for collective learning
## Note
This is in Python 3.5.
Requirements: in requirements.txt + OpenCV

## Setting up
### Set up virtualenv
http://python-guide-pt-br.readthedocs.io/en/latest/dev/virtualenvs/

### Download required packages
Activate your virtual environment, then issue this command:
```
pip install -r requirements.txt
```

### Change euclid.py file
Euclid has some syntax that doesn't work in Python 3, so you need to change the file in the installed packages.
Go to your virtualenv folder > lib > python 3.5 > site-packages. Then replace the euclid.py file there with the one in this repo.

### Set up IPython kernel to work with the virtualenv
Make sure your virtualenv is activated. Then issue the following commands:
```
(env)$ pip install jupyter
(env)$ pip install ipykernel
(env)$ python -m ipykernel install --user --name testenv --display-name "Python3 (dqn-virtualenv)"
```
## Files
### Jupyter notebooks
#### training


#### testing_5preds

#### testing_independents

#### analysis

### Python files
#### tf_rl/simulate.py

#### tf_rl/simulation/karpathy_game.py

## Issues
1. Inactive predators cannot eat
2. There can only be one physical cue (but several cue types)
