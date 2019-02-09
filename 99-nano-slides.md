Overview
Trilogy of trilogy (like Star Wars)

Part 1: Math, Python, tools, concepts
    a) Matrix
    b) Function
    c) Statistics and Prob

    Python: install, language, images/lists/numpy

Part 2: Neural networks
    APIs: tensorflow, keras, pytorch
    Image classifier

Part 3:

---
Motivation for session #4
* Tensorflow is a computational graph!
* GPU (very very fast matrix multiplication)

Session #4
Intro to tensorflow: build a computational graph
Intro to NN using just numpy

Overview:

#4: intro to nn using numpy, intro to tensorflow, intro to panda (speed in convolution)

#5: nn using tensorflow api, nn using pytorch
#6: keras, different types of convolve ops

----

List of Quotes:

What's the Matrix?
A picture is worth a thousand words

If a tree falls in a forest and nobody's around to hear it, does it make a sound?

A line is a dot that went for a walk
- Paul Klee


I have a dream
  -Martin Luther King

https://deepdreamgenerator.com/



## Session 1 - Images - Topics

    1. Math : Matrix Multiplication
    2. Python
        * Installation (conda, ipython)
        * Image Manipulation (read/process/write files)
    3. ML: ml5js.org 
        * Image classifier
        * PoseNet
        * Style Transfer

## Session 2: magenta.js

    Python
        - Jupyter notebooks (whirldwind tour, share n collaborate)
        - numpy -> dot product, convolutions, downsampling!
    
    Images
        Downsampling

    Math
        - Function, polynomial: linear, quadratic, cubic (smooth fns)
        - Plotting functions using matplotlib
        - Find (maxima and) minima of a quadratic function

    ML: magenta.js (music)
        https://experiments.withgoogle.com/ai/melody-mixer/view/
        https://github.com/googlecreativelab/melody-mixer

        https://medium.com/@oluwafunmi.ojo/getting-started-with-magenta-js-e7ffbcb64c21
        https://hello-magenta.glitch.me/


## Session 2 - detailed

    Numpy
        np.array(), np.zeros(), np.exp(), np.dot() vs np.matmul()
        np.random.random(), randint(), sample(), choice(), shuffle()
        from list to list

        - Find maxima and minima of a quadratic function
            -> Create sample data -> find weights/params
            -> Create noisy data with random
        - Downsampling an image (pick max, avg?), more image filters with numpy (implement kernels)

    a) How to run WhirlWind Tour of Python locally using jupyter notebook

    Python lib: numpy
    ML: web app using Flask to input an image and make a prediction!
    And deploy to now.js?

    Math: functions and plotting! plot functions using matplotlib
    How can we approximate a function value? Given many X, Y

    ML: magenta projects and APIs!?

    Downsampling images: by half and stride=2 (MaxPooling!)

Quadratic function: find minima, maxima programmatically

Have some error function -> how close we are and tweak...


Model: quadratic with some known params
-> create test data -> predict the weights/parameters
-> iterate and improve the prediction



Potential projects using ml5js
a) Create a web front end to pick images and make predictions!
b) Create a web front end for picking image style transfer (select different artist mode, source)
c) PoseNet: project Sombrero or Robotize! Or Anonymize


# Session 3: tensorflow.js, 

    Demo: Kodelife Shaders


    Numpy -> Tensorflow? Why??? GPU!

    * GPU
        Video cards... gpu processors (graphics, other matrix mult)
        CPU
        GPU
            Nvidia -> CUDA drivers <- tensorflow!
            AMD

    Video! Melt!!! Seriously.js (webrtc.....)

    Python:
        Pandas

    https://meowni.ca/posts/hello-tensorflow/

    https://js.tensorflow.org/tutorials/fit-curve.html

    A. Statistics: averaging faces, stdev of faces! where change happens the most!
    celeb database -> grab first 20!

        Vertical line detector -> given an image
        Horizontal line detector, slanted, curved feature (times 4)

    Image feature detection
        -> Dot product of two similar matrices?
        -> Mouse detect: curve + straight?

    B. Neural Network in 11 lines using numpy!

    -> PoseNet -> canvas (to add a sombrero), parrot on right shoulder


Translation to Tensorflow.js (And tensorflow in Python ???)


PyTorch tutorial
11 lines of NN to PyTorch
http://iamtrask.github.io/2017/01/15/pytorch-tutorial/


# Session 4: ForwardFeed NN

    A. Intro to tensorflow, tensorflowjs and curve fitting (using tensorflowjs)

# Session 5: RNN

# Session 6: ConvNet
    Object Detection
    Segmentation

# Session 7: LSTM
    Sentiment Analysis

Machine Learning Topics
    * k-clustering
    * Recommenders

# Session 8: GANs


# Session 9:




### Running Python scripts

    $ python <filename>

    $ ipython

    [1] odds = [n for n in range(10) if n % 2 == 1]
    [2] print("The first odds: ", odds)
    [3] %save odds.py 1-2
    [4] quit()

    $ python odds.py

    # Run a Python web server
    # Python 2.7
    $ python -m SimpleHTTPServer

    # Python 3
    $ python -m http.server 8080







====


Extras (not that impressive for my needs!!)
#### Example: Draw a gray cross over an image

    from PIL import Image, ImageDraw

    image = Image.open("panda-corner.jpg")
    image.show()

    draw = ImageDraw.Draw(image)
    draw.line((0, 0) + image.size, fill=128)
    draw.line((0, image.size[1], image.size[0], 0), fill=128)
    del draw

    # write to stdout
    image.save("panda-cross.png", format="PNG")



#### RGBA -> Create transforms using RGBA!!!

    print(image.format) # Output: JPEG
    print(image.mode) # Output: RGB
    print(image.palette) # Output: None


TODO: Make sure the convolution kernel can work with numpy arrays or transforms easily.

