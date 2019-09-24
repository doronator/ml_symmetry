# ml_symmetry
Research on machine learning and symmetry

## Online materials to accompany the paper "Symmetry constrained machine learning"

The computational experiments described in the paper can all be reproduced using 
the code in this repository, in particular by running the jupyter notebook.

Make sure your python installation supports jupyter, and that you have installed 
the python libraries in the file requirements.txt, and you can start the jupyter 
notebook by running the following command from the home directory of this repository:
`jupyter notebook`

We explore the idea of imposing precise symmetries on a machine learning model. 
We propose a conceptually straightforward and general approach to achieve this goal. 
We use a toy problem to demonstrate how this approach can be easily implemented, 
and point out the dangers that can arise from not being careful about imposing 
symmetry on to a machine learning model.

We argue that considering symmetry in the context of machine learning can expose 
weaknesses in machine learning systems, and can be used to guard against these
weaknesses. Furthermore, symmetry can help make machine learning models more 
compact, and robust to over-fitting, with the ability to satisfy symmetry 
conditions with complete certainty.

[![Build Status](https://travis-ci.org/doronator/ml_symmetry.svg?branch=jmlr)](https://travis-ci.org/doronator/ml_symmetry)
