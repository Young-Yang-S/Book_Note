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

*  How to create Array   

  (1) **Create from list** 
```
    np.array([1, 2, 3, 4], dtype='float32')
    np.array([3.14, 4, 2, 3])
```
        For ana array, the element should have the same type (like string, int, float and etc)
     
  (2) **Create ftom built-in method**
```
    np.zeros(10, dtype=int)              # Create a length-10 integer array filled with zeros
    np.ones((3, 5), dtype=float)        # Create a 3x5 floating-point array filled with 1s
    np.full((3, 5), 3.14)                      # Create a 3x5 array filled with 3.14
    np.arange(0, 20, 2)                     # Create an array filled with a linear sequence, Starting at 0, ending at 20, stepping by 2
    np.linspace(0, 1, 5)                     # Create an array of five values evenly spaced between 0 and 1
    np.random.random((3, 3))          # Create a 3x3 array of uniformly distributed random values between 0 and 1
    np.random.normal(0, 1, (3, 3))   # Create a 3x3 array of normally distributed random values with mean 0 and standard deviation 1
    np.random.randint(0, 10, (3, 3))  # # Create a 3x3 array of random integers in the interval [0, 10)
    np.eye(3)                                     # Create a 3x3 identity matrix  (Diagonal element is 1)
```
*  Numpy data type and how to express it:
    *  np.zeros(10, dtype='int16')  # use quote
    *  np.zeros(10, dtype=np.int16)  # use np.

   ![image](https://user-images.githubusercontent.com/76230868/118315780-f0697e80-b4c3-11eb-8010-6cf5ca84b934.png)
   
*   Basic numpy array manipulation
    **Attributes of Array, Indexing of Array, Slicing of Arrays, Reshaping of Arrays, Joining and Splitting of Arrays**
 
 *   Attributes of Array
     np.random.seed(0) # set seed for reproducibility
     x3 = np.random.randint(10, size=(3, 4, 5)) # Create a three-dimensional array
     
     *  x3 ndim: 3               # the number of dimensions
     *  x3 shape: (3, 4, 5)   # the size of each dimension
     *  x3 size: 60               # the total size of the array
     *  x3.dtype:                 # data type
     *  x3.itemsize              # each item size in bytes 
     *   x3.nbyte                  # total size of array in bytes
*   Indexing 
    *  Exampe Array:  x1 = array([5, 0, 3, 3, 7, 9])
        x1[4] = 7 
        x1[-1] =  9
    *  Exampe Array:   x2 = array([[3, 5, 2, 4], [7, 6, 8, 8],[1, 6, 7, 7]])
        x2[2, -1] = 7
        
     * Change the value of item in array
       x2[0,2] = 12 

*    Slicing
(1)  **One dimensional array** 
      *  x[:5]         # first five elements
      *  x[5:]        # elements after index 5
      *  x[4:7]      # element from forth to seventh
      *  x[1::2]     # starting at index 1,step is 2,  like array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), return  array([1, 3, 5, 7, 9])
      *  x[::-1]      # all elements, reversed
(2)  **Two dimensional array**
      *   Example: array([[12, 5, 2, 4],
                                     [ 7, 6, 8, 8],
                                     [ 1, 6, 7, 7]])
      *   x2[:2, :3]        # two rows, three columns
      *   x2[:3, ::2]       # three rows, column step is 2
      *   x2[::-1, ::-1]    # reverse array
      *   x2[:, 0]           # get all rows, first column
     **Tip** for slicing: in numpy, all of these slicing method return the view of array rather than the copy of array which means if you change the returned sub array, then the original array will also be changed
 
*    Get the copy of array: 
```
x2[:2, :2].copy()
```

* Reshaping of array:

(1)  Change the shape of array:
       np.arange(1, 10).reshape((3, 3))      # change to 3,3 dimensions
 
(2)  Change the dimension of array:
       x.reshape((1, 3))          # change the one dimensional array to two dimensioanl (1,3) array
 
 *    Concatenation
      np.concatenate([x, y, z]   # concatenate three arrays, concatenation doesn't change the dimension of array
      
      More userful way: **vstack, hstack**
      
      np.vstack([x, grid])
      np.hstack([grid, y])
  
  * Splitting
  
(1) Splitting one dimensional array 
    x = [1, 2, 3, 99, 99, 3, 2, 1]
       
   x1, x2, x3 = np.split(x, [3, 5])        # split from index 3 and 5       [1 2 3] [99 99] [3 2 1]
   
 (2)  vsplit and hsplit
        *   array([[ 0, 1, 2, 3],
                     [ 4, 5, 6, 7],
                     [ 8, 9, 10, 11],
                     [12, 13, 14, 15]])
                     
          np.vsplit(grid, [2])                    # split horizontally from index 2                    
           [[0 1 2 3]                   [[ 8 9 10 11]               
            [4 5 6 7]]                   [12 13 14 15]]
            
            np.hsplit(grid, [2])                   # split vertically from index 2 
            [[ 0 1]
            [ 4 5]
            [ 8 9]
            [12 13]]
           
           [[ 2 3]
            [ 6 7]
            [10 11]
            [14 15]]
            
         
*    Array arithmetic 

    *  **x = [0 1 2 3]**
     
      x + 5 = [5 6 7 8]
      x - 5 = [-5 -4 -3 -2]
      x * 2 = [0 2 4 6]
      x / 2 = [ 0. 0.5 1. 1.5]
      x // 2 = [0 0 1 1]
       x = [ 0 -1 -2 -3]
       x ** 2 = [0 1 4 9]
       x % 2 = [0 1 0 1]
  
![image](https://user-images.githubusercontent.com/76230868/118343547-b87c2e80-b4f7-11eb-8f93-213f029ca619.png)

*     Absolute  value
      abs(x)
 
*    Trigonometric functions
      * np.sin(theta)
      * np.cos(theta)
      * np.tan(theta)
      * np.arcsin(x)
      * np.arccos(x)
      * np.arctan(x)
      
*     Exponents and logarithms
        * np.exp(x)             # e^x
         * np.exp2(x)          # 2^x
         * np.power(3, x)    # 3^x
         * np.log(x)
        *  np.log2(x)
         * np.log10(x)
   
*   The above operations are the common one in numpy, there are a lot of others which can be checked in documentation

*    Aggregation 1
     * np.add.reduce(x)     # calculate the sum of items in array
     * np.multiply.reduce(x)  # calculate the product of items in array, we can apply reduce function after arithmetic operators
     * np.add.accumulate(x)  # accumulate is to show all the intermediate results of computations
     * np.multiply.accumulate(x)
        
*    Aggregation 2

(1)  One dimension:
      * np.sum(L)        # calculate the sum of items in array
      * np.min(big_array), np.max(big_array)        # get the min and max 
      * heights.mean()
      * heights.std()
      * np.percentile(heights, 25))
      *  np.median(heights))
      *  np.percentile(heights, 75))
      
   Or big_array.min(), big_array.max(), big_array.sum()  use this syntax

(2)  Multidimensional 
       * M.sum()      # sum all
       * M.min(axis=0)  # sum by column
       * M.min(axis=1)   # sum by row 
       

![image](https://user-images.githubusercontent.com/76230868/118344058-4eb15400-b4fa-11eb-8d2c-80138d131542.png)


*  Broadcasting
   
   This is used to solve the problem of addition, substraction, multiplication etc on arrays of **different sizes**
   
   a = np.array([0, 1, 2])
   
   M = np.ones((3, 3))
 
   M + a =  array([[ 1., 2., 3.],
                             [ 1., 2., 3.],
                             [ 1., 2., 3.]])
        
  ![image](https://user-images.githubusercontent.com/76230868/118345678-29c1de80-b504-11eb-88a7-0983d1f8ea6c.png)


* Boolean operation

   x = np.array([1, 2, 3, 4, 5])

    x == 3
    
    The return of it is  array([False, False, True, False, False], dtype=bool)
    
    ![image](https://user-images.githubusercontent.com/76230868/118345895-bcaf4880-b505-11eb-878e-f0ea4c491654.png)

*  compound expressions means like this  (2 * x) == (x ** 2)

*  Boolean Function 
   
   With boolean array, we can do a lot of stuff,  like set some conditions and return certain element.
   
   np.count_nonzero(x < 6)  # how many values less than 6?
   
    np.sum(x < 6, axis=1)    # how many values less than 6 in each row?
    
    np.any(x > 8)   # are there any values greater than 8?
    
    np.all(x < 10)  #  are all values less than 10?
    
    np.all(x < 8, axis=1)   # are all values in each row less than 8?
    
    x[x < 5]    # return item which is less than 5
    
  *  Boolean operator 
  
     np.sum((inches > 0.5) & (inches < 1))
     
     ![image](https://user-images.githubusercontent.com/76230868/118346104-3267e400-b507-11eb-976d-49d5b3939783.png)

  *  Generate random array
 ```
 rand = np.random.RandomState(42)
 x = rand.randint(100, size=10)

 [51 92 14 71 60 20 82 86 74 74]
 ```
 
 *  Fancy index (Get the subset of array, modify certain values in array)

     (1)  Subset of array
  
    One good stuff for fancy index is that it can change the dimension of result that we desire
    
    * One dimension
 
     x = [51 92 14 71 60 20 82 86 74 74]
     
     ind = [3, 7, 4]
     
     x[ind]
     
     array([71, 86, 60])
     
     ind = np.array([[3, 7],
                            [4, 5]])
                           
     x[ind]
     
     array([[71, 86],
                [60, 20]])
                
     * Two dimension 

        For two dimension array, we should use two array to get the value of it
        
        X = array([[ 0, 1, 2, 3],
                     [ 4, 5, 6, 7],
                     [ 8, 9, 10, 11]])
         
         row = np.array([0, 1, 2])
         col = np.array([2, 1, 3])
         X[row, col]
         
         
         Or more complex one:
              X[2, [2, 0, 1]]       # get the third, first, second item in second row
              
          X[1:, [2, 0, 1]]        # get the third, first, second item after second row
          
         (2)  Modify values with fancy indexing 
          
         x = np.arange(10)
         
          i = np.array([2, 1, 8, 4])
          
          x[i] = 99
          
          x[i] -= 10
             
      *  Generate two-dimensional normal distribution
         
          mean = [0, 0]
          
          cov = [[1, 2],
                    [2, 5]]
                    
          X = rand.multivariate_normal(mean, cov, 100)
     
**Two dimensional normal distribution**
![image](https://user-images.githubusercontent.com/76230868/118346674-31d14c80-b50b-11eb-881c-8c0292d1383a.png)

*  How to select certain random number 
    
    The example 2-D array is the 2-D normal distribution points in X in the last star point
    
     **Fristly, we choose random index of item of 20 from number of X.shape[0] total items
    
     indices = np.random.choice(X.shape[0], 20, replace=False)
     
     **The we use fancy indexing to get the data
     
     selection = X[indices]
     
     This is a useful method for splitting training data set and test data set
     
     **Twenty random points from 100 points**
     
     ![image](https://user-images.githubusercontent.com/76230868/118346836-5e399880-b50c-11eb-8202-8515ded98c3c.png)

*  Sort function
   
   np.sort(x)  or  x.sort()   # sort ascending
   
   np.argsort(x)    # sort and get the index
   
   We can use fancy indexing method to use result of argsort to get the item sort result
   
   np.sort(X, axis=0)   # sort each column of X
   
   np.sort(X, axis=1)  # sort each row of X
   
   
   
   
     
     
      
    
    



     
  
  
   
   
   

