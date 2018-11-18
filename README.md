# Notebooks

This is a collection of various python notebooks I use for research. I'll try to keep this readme relatively up to date but you know how things go

The top level notebooks are usually early prototypes that aren't developed enough to justify their own folder yet

* dm_control - some work to interface with deepminds control suite
* drake_examples - various examples using drake I've made or that I've gotten from the underactuated course notes
* walkers - trajectory optimization for planar walkers using drake, also includes playback in mujoco
* robosimian - same idea as walkers but for the robosimian (this is much more a work in progress)
* misc - miscellaneous scratch work

# Various libraries I use (that you will need to install)

## Mujoco/Mujoco py
http://www.mujoco.org
https://github.com/openai/mujoco-py (also has instructions on how to install mujoco)

Note you need a liscense to use mujoco, but a free one can be obtained without too much pain. either get a no questions asked 30 day trial or a free one year renewable one. If you publish with it you will need to pay up though!

## Open AI baselines
https://github.com/openai/baselines

Note: Make sure you install from the github and not the version that you can obtain by default using pip

## Lots of others that can be installed with pip

Just use pip -r install requirements.txt




## Drake
I'm not actively using drake, but if you want to run/use some of my earlier examples then you will need to install drake too

(http://drake.mit.edu). As well as the code/notes here 
(https://github.com/RussTedrake/underactuated). Both will need to be added to 
your python path, it should look something like

export PYTHONPATH=path_to_drake/lib/python2.7/site-packages:path_to_underactuated/underactuated/src/:${PYTHONPATH}
