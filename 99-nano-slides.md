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

    scikit-learn !?

    https://github.com/scikit-learn/scikit-learn
    https://scikit-learn.org/stable/auto_examples/index.html

    https://github.com/PacktPublishing/scikit-learn-Cookbook-Second-Edition


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



### Run a Python web server

    # Python 2.7
    $ conda activate python2.7
    $ python --version
    $ echo '<h1>Hello, Python 2.7!</h1>' > index.html
    $ python -m SimpleHTTPServer

    # Python 3
    $ conda activate nanos
    $ python --version
    $ echo '<h1>Hello, Python 3!</h1>' > index.html
    $ python -m http.server 8080

====

#### RGBA -> Create transforms using RGBA!!!

    print(image.format) # Output: JPEG
    print(image.mode) # Output: RGB
    print(image.palette) # Output: None


TODO: Make sure the convolution kernel can work with numpy arrays or transforms easily.


http://web.cs.ucdavis.edu/~yjlee/teaching/ecs289g-winter2018/Pytorch_Tutorial.pdf

-----

Nano sessions

#1 Python (anacaonda, ipython)
    Matrices -> Image processing (greying, compressing/downsampling)

#2 Numpy
    Functions -> Plotting, approx
    -> 
    Lambdas

#3 Pandas
    Stats -> Averaging faces, Stdev of faces
    and Prob? (Bayes's theorem)?
    Naive Bayes -> Linear estimator?
    Recursion / Tree


Tensorflow Notes
- Slides from cs20: http://web.stanford.edu/class/cs20si/syllabus.html

Fun notes!!
    07_ConvNets in tensorflow -> Awesome explanation of ConvNets
    Conv NN -> Style Transfer


Of the many ML topics, the most fundamentals:

1) Classification: image classifier using MNIST

Linear Regression
SVM: support vector machine

Fun topics:
DL

CNN: Computer Vision


RNN: Sequence Models, Language/Sentence
    vec2words


ML in 4 sessions
#4 Classification
    Image classifier

#5 k-clustering?



#6 Tensorflow and Neural Networks


    -> Model Zoo NN

#7 ML map (this could be a nice interactive app)

Pre-trained Models? https://modelzoo.co/

https://github.com/BVLC/caffe/wiki/Model-Zoo
https://github.com/tensorflow/models
https://github.com/pytorch/vision/tree/master/torchvision/models

Keras slides!

AI
    * ML
        -Linear Regression
        -SVM
        -Trees/Ensemble
        -K-clustering

        * DL
            NN: perceptron
                * Not new, but it requires:
                    -> Massive amounts of data (Internet!)
                    -> Massive computing power (GPUs)
            CNN: ConvNet. Computer Vision
                Model: VGG16, ResNet, AlexNet, LeNet, Inception, Yolo
                - Face recognition
                - Pose recognition
                - Semantic Segmentation / Instance Segmentation
                - Self-driving cards (Detection)
                - Deep Dreams
                - Style Transfer
                - Colorization (old photos, line drawings)
            RNN (sequence model, based on time)
                RNN, GRU, LSTM, word2vec, wavenet
                - Language
                    - sentiment analysis
                    - speech recognition
                    - translation
                    - text summarization
                    - Chat bots
                        Duplex restaurant reservation
                - Music
                    - jazz improvisation
                    - jamming duet
                - Image captioning
    
            GANS (Adversarial Networks)
                Deep Fakes?

            RL (Reinforcement Learning)
                - AlphaGo beating top Go player
                - Atari AI players
                - Dota 2 AI players
                DQN: Deep Q Networks


In God we trust; others must provide data
Anonymous

Popular Datasets
    MNIST
        http://yann.lecun.com/exdb/mnist/
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    CIFAR: http://www.cs.toronto.edu/~kriz/cifar.html
            
    Celebrity Database

    Kaggle Datasets

Frameworks
    Tensorflow
        google
        stanford cs20
    Keras
    Scikit
    PyTorch (FB)
        udacity
    Caffe/Caffe2
    Theano
    MXNET
    AWS?

    Companies:
        Uber - Horovod: https://eng.uber.com/horovod/





Roadmaps to learn?

Data Scientist Roadmap
http://nirvacana.com/thoughts/2013/07/08/becoming-a-data-scientist/

https://machinelearningmastery.com/machine-learning-for-programmers/

https://machinelearningmastery.com/process-for-working-through-machine-learning-problems/

Small data sets UCI
https://machinelearningmastery.com/practice-machine-learning-with-small-in-memory-datasets-from-the-uci-machine-learning-repository/


Weka:
https://machinelearningmastery.com/how-to-run-your-first-classifier-in-weka/
https://www.cs.waikato.ac.nz/ml/weka/book.html

Data sets:
    http://deeplearning.net/datasets/
    https://www.kaggle.com/datasets

    CIFAR10
    MNIST
    ImageNet
    MovieLens
    Celebs (mscoco?)
    Iris Data Set
        https://en.wikipedia.org/wiki/Iris_flower_data_set

Celebs, 200,000 images of celebrities
http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html


Confusion matrix
----

Plus interactive MNIST (using keras.js)

----

Present MNIST classifier using scikit-learn (from the book)

but also show the equivalent code in tensorflow.js for making predictions
-> contrast the two

also the version using keras API

////------ Fun projects

-> PoseNet (use the data points to camouflage the impersonator)

Sombrero contest
-> Add a black sombrero to the red clown, red sombrero to the black clown


Projects
--------

a) Sombrero project?
b) Style Transfer -> Deep Dream Generator

c) Music with magenta.js


Tutorials
http://openbookproject.net/thinkcs/python/english3e/index.html


CIFAR10 Demo (WoW!)
https://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html


while True:
    data_batch = dataset.sample_data_batch()
    loss = network.forward(data_batch)
    dx = network.backward()
    x += learning_rate * dx



Dropout (in a 3-layer network)

p = 0.5

def tran_step(X):
    H1 = np.maximum(0, np.dot(W1, X) + b1) # Relu Activation
    U1 = np.random.rand(*H1.shape) < p # first dropout mask
    H1 *= U1 # drop

    H2 = np.maximum(0, np.dot(W2, H1) + b2)
    U2 = np.random.rand(*H2.shape) < p # first dropout mask
    H2 *= U2 # drop
    out = np.dot(W3, H2) + b3

def predict(X): # p: dropout rate at test time
    H1 = np.maximum(0, np.dot(W1, X) + b1) * p
    H2 = np.maximum(0, np.dot(W2, H1) + b2) * p
    out = np.dot(W3, H2) + b3



CNNs

https://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html


=====

Killing pandas 
 -> not used in any ML
 -> only useful for data exploration

### Pandas
    1. Read a CSV file and display
    2. Select

    https://pandas.pydata.org/pandas-docs/stable/user_guide/cookbook.html
    http://pandas.pydata.org/Pandas_Cheat_Sheet.pdf

    https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html

====

Deep Training workflow

A) Train the Model
    a) Prepare the data (normalize it)
    b) Train the model
        -> keep reducing the error loss function to determine the weights of W

    c) Test the model
        With the given determined W, make predictions on the test set
        Compute the error we get here

    d) Compare the error in the training vs the test
        -> large gap: overfitting the data? decrease model complexity
        -> small gap: maybe underfit? increase model complexity

B) Make predictions
    Given the weights produced during training,
    make predictions in the real world

    Use the pre-trained model (check model zoo) to make predictions

A) Computationally expensive, it can take days or weeks
B) Predictions should be relatively fast to compute from the model and weights.


Intro to Parallel Programming
https://classroom.udacity.com/courses/cs344/
https://www.youtube.com/playlist?list=PLAwxTw4SYaPnFKojVQrmyOGFCqHTxfdv2


http://ufldl.stanford.edu/

----

RNN

-> hello ?
---> English sonnet? Shakespeare?

    * Open source math topology
        -> Diagrams and Lemmas

    * Linux source
        -> C source code + comments (no-sense)

=> Training on character sequence

