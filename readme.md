# HillClimber

HillClimber is an implementation of [HillClimbing](https://en.wikipedia.org/wiki/Hill_climbing) algorithm
to optimize the weights of a given neural network.

This repository also contains:

* A [Dash](https://plotly.com/dash/) application to visualize the searching process
and the search graph (partially, since the graph has more than 3 dimensions).
* A Jupyter notebook to demonstrate how to use the packages.
* Sample log and model files.

### [GymClimber ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/katipber/hillclimber/GymClimber.ipynb)

## Tech / Frameworks
* OpenAI Gym
* PyTorch
* Plotly Dash
* Numpy
* Pandas
* SciPy

## Screenshots

| CartPole-v1  | MountainCar-v0 | LunarLander-v2|
|     :---:    |     :---:      |     :---:     |
| ![CartPole](sample/media/CP.gif?raw=true) | ![MountainCar](sample/media/MC.gif?raw=true) | ![LunarLander](sample/media/LL.gif?raw=true) |
| ~10-30 seconds | ~1-10 minutes  | ~10-60 minutes |

<br>

##### CartPole Score Graph
![CartPole Score](sample/media/CP_score.jpg?raw=true)

<br>

##### MountainCar Search Graph
![MountainCar Search](sample/media/MC_search.jpg?raw=true)

<br>

##### LunarLander Search Graph
![LunarLander Search](sample/media/LL_search.jpg?raw=true)


## Credits
I was inspired by this [video](https://www.youtube.com/watch?v=WZFj81xPgyk&list=PLIfPjWrv526bMF8_vx9BqWjec-F-g-lQO&index=6) of [TheComputerScientist](https://www.youtube.com/channel/UCUbeqkIVP808fEP0ae-_akw).
It is a nice channel if you are interested in AI.

Also [recorder.py](recorder.py) is taken from [highway-env](https://github.com/eleurent/highway-env).
I was planning to include highway environment in this project as well,
but for some reason each episode took around 3-12 seconds during my trials,
so I decided not to.
