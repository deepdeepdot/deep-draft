## What's the Matrix?

![The Matrix Image](../img/matrix_reboot.jpg)


### Session 1

A line is a dot that went for a walk
- Paul Klee


### Topics

    1. Math : Matrix Transpose and Multiplication
    2. Python
        * Installation (conda, ipython)
        * Image Manipulation (read/process/write files)
    3. ML: ml5js.org 
        * Image classifier
        * PoseNet
        * Style Transfer


### A. Intro to Python


### What's Python?
* High-level language for science and cs learning
    * Web development.
        http://flask.pocoo.org/
    * Game development.
        http://inventwithpython.com/pygame/
    * Data science and machine learning!
        https://scikit-learn.org/
    * Music with Python! http://foxdot.org


### Setup Anaconda

    # Version Manager for Python and package manager
    # Anaconda is like rvm in Ruby or nvm in node

    # Install anaconda 3.7. https://www.anaconda.com/distribution/
    # Popular Python versions: 2.7, 3.6 and 3.7

    $ python --version
    $ conda env list
    $ conda create -n nanohackers python=3.6
    $ conda activate nanohackers
    $ python --version

    # Exercise
    # Create a conda environment named 'python2.7' having python v2.7


### Running Interactive Python

    $ ipython

    [1]: print("Hello world!")

    [2]: def add(first, second):
            return first + second

    [3]: print("Total:", add(3, 5))

    [4]: help(print)  # press 'q' to quit help
    [5]: ?print

    [6]: help(add) # press 'q' to quit help

    [7]: quit()


### Python List
    [1]: numbers = [1, 2, 3, 4, 5]
    [2]: numbers

    [3]: names = ['george', 'donald', 'obama']
    [4]: print(names, len(names))

    [5]: for prez in names:
            print("One prez: ", prez)

    [6]: print([prez for prez in names])


### List slices
    [1]: numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    [2]: numbers[0]

    [3]: numbers[3:7]

    [4]: numbers[0:9:2]
    [5]: numbers[0:9:3]

    [6]: multiply_of_3 = numbers[0:9:3]


### List Comprehension
    [1]: numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    [2]: odds = [n for n in numbers if n % 2 == 1]
    [3]: odds # same as print(odds)

    [4]: triple_odds = [3 * n for n in numbers if n % 2 == 1]

    [5]: dozen = [n for n in range(12)]

    [6]: # Double Array of 3 rows and 4 columns
    [6]: double_array = [[i*p+1 for i in range(4)] for p in range(3)]
    [7]: print(len(double_array[0]), "rows x",
               len(double_array), "columns")


### Challenges

    1. List of squares of the first 7 numbers using list comprehension

    2. Given a list of president names, return the list of presidents
    that contain the letter 'h' either in the first name or last name

    Complete the list using wikipedia
    https://en.wikipedia.org/wiki/List_of_Presidents_of_the_United_States

    presidents = ["george, washington", "john, adams", "thomas, jefferson"]

    3. Retrieve the first names of all the president names
    Hint: look for 'find' and 'split' from:
    https://docs.python.org/2/library/string.html#string-functions

    4. Retrieve the list of presidents in which the first name has an 'h'


### Reference

* Whirlwind Tour of Python<br/>
https://nbviewer.jupyter.org/github/jakevdp/WhirlwindTourOfPython/blob/master/Index.ipynb

* List Comprehension<br/>
https://treyhunner.com/2015/12/python-list-comprehensions-now-in-color/

* Python Practice!!!<br/>
https://codingbat.com/python


### B. Math - Matrix


### Matrix Transpose

    A = [[0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11]]

    tranpose(A) = [
        [0, 4, 8],
        [1, 5, 9],
        [2, 6, 10],
        [3, 7, 11]
    ]

Python challenge: Implement transpose()


### Matrix Multiplication

http://matrixmultiplication.xyz/

    X = A * B

    X[i,j] = Sum(A[i,k] * B[k,j]) over all i,j,k

    A: m x n (m rows by n columns)
    B: n x p (n rows by p columns)
    X: m x p (m rows by p columns)


### Python Challenge!

Implement Matrix Multiplication using Python

    def multiply(A, B):
        None
    
    A = [[1, 2, 3, 8],
        [2, 0, 3, 9],
        [0, 1, 3, 1]]

    B = [[3, 1], [4, 4], [6, 5], [2, 0]]

    print(len(A), len(A[0]))
    print(len(B), len(B[0]))

    X = multiply(A, B)


### C. Images with Python


### Install the pillow library

    $ conda activate nanohackers
    $ conda list
    $ conda list | grep pillow
    $ conda install pillow
    $ conda list | grep pillow


### RGB/RGBA Color Model

    An Image consists of 4 channels:
        * Red    * Green
        * Blue   * Alpha (optional)

    An RGB image of 200x200 pixels contains 
    3 layers of 200x200 values.

    An RGBA image of 200x200 pixels contains 
    4 layers of 200x200 values.

    Each pixel value goes from 0 to 255

    * How many color combinations can we achieve in a pixel?
    * How many pixels can we have in a 5 mpx, 12 mpx, 24 mpx?
      What would be the width/height of such images?


#### Ex: Create a greyscale image

    import matplotlib.pyplot as plt

    img = plt.imread("panda-corner.jpg")
    nrows, ncols = img.shape[0], img.shape[1]

    copy = img.copy()

    for row in range(nrows):
        for col in range(ncols):
            avg = sum(img[row,col,:]) / 3
            copy[row,col,:] = avg

    plt.imshow(copy), plt.show()
    plt.imsave("panda-grey.png", copy)


#### Ex: Image Compression

<p align="left">Suppose you have an image of size 640x360, how can we get an image of size 320x180 (half the image!?)</p>

    import matplotlib.pyplot as plt

    img = plt.imread("panda-corner.jpg")
    nrows, ncols = int(img.shape[0]/2), int(img.shape[1]/2)

    smaller = img[0:nrows, 0:ncols/2, :]
    plt.imshow(smaller), plt.show()

    img.shape, smaller.shape

What should be the value of each pixel???


### Downsampling

<!-- (https://adeshpande3.github.io/assets/MaxPool.png) -->
![MaxPooling Image](../img/maxpool.png)

Python Challenge: implement image compression by downsampling!


#### Image Kernels and convolutions

http://setosa.io/ev/image-kernels/

    # Convolution
    # https://docs.gimp.org/2.8/en/plug-in-convmatrix.html

    # Python Challenge: implement image kernel!
    # More things to try out

    1. Process only for a small section (not the full image)
    2. Process the inverse (all the areas BUT the selected area)
    3. Use a different mask for the selected area! Say some rabbit!
    4. Try out some colormap ("hot"?)


### Reference

* Python: https://nbviewer.jupyter.org/github/jakevdp/WhirlwindTourOfPython/blob/master/Index.ipynb
* Matrix:
    https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/linear_algebra.html

* Pillow: https://matplotlib.org/users/image_tutorial.html
* Pixels: https://processing.org/tutorials/pixels/
* RGB: https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_Colors/Color_picker_tool


### D. ML Projects


### ml5js
https://ml5js.org/

    * Image classifier
    * PoseNet
    * Style Transfer

    Exercise
    Extend the Image classifier to train and recognize your own images


### Credits
* Neo, Morpheus and Trinity
* https://computersciencewiki.org/index.php/Max-pooling_/_Pooling
