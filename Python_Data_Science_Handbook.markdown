# Python Data Science Handbook 
 
by **Jake VanderPlas**  (Follow the step of Jake and explore the DS field by Python)

## Module One: Preface 

*  Data science is an interdisciplinary subject

*  Three key distinct areas which together comprise data science: **Statistics, Computer Science and Domain expertise**.  A statistician who knows how to model and summarize
datasets; A computer scientist who can design and use algorithms to efficiently store, process, and visualize this data; and A domain expertise—necessary both to formulate the right questions and to put their answers in context.

*  Python packages Data scientist must know: **NumPy** for manipulation of homogeneous array-based data, **Pandas** for manipulation of heterogeneous and labeled data, **SciPy** for common scientific computing tasks, **Matplotlib** for publication-quality visualizations, **IPython** for interactive execution and sharing of code, **Scikit-Learn** for machine learning.

## Module Two: Numpy Introduction

*  Image can be regarded as two-dimensional array of numbers representing pixel brightness across the area.

*  The first key thing for data scientist is to convert the information (Image, Text, Sound) we are focusing into arrays of data.

*  How to import Numpy module: Import numpy as np

*  **How to create Array**   

(1) **Create from list** 
```
    np.array([1, 2, 3, 4], dtype='float32')
    np.array([3.14, 4, 2, 3])
```
        For ana array, the element should have the same type (like string, int, float and etc)
(2) **Create ftom built-in method**
```
    np.zeros(10, dtype=int)          # Create a length-10 integer array filled with zeros
    np.ones((3, 5), dtype=float)    # Create a 3x5 floating-point array filled with 1s
    np.full((3, 5), 3.14)                  # Create a 3x5 array filled with 3.14
    np.arange(0, 20, 2)                 # Create an array filled with a linear sequence, Starting at 0, ending at 20, stepping by 2
    np.linspace(0, 1, 5)                 # Create an array of five values evenly spaced between 0 and 1
    
```
