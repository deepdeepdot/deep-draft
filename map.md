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
    
            Generative Models:
                Autoencoders and Variational Autoencoders
                GANS (Adversarial Networks)
                    DeepDream
                    SuperResolution

            RL (Reinforcement Learning)
                - Deep Q Networks (DQN)
                - AlphaGo beating top Go player
                - Atari AI players
                - Dota 2 AI players
                DQN: Deep Q Networks
            
            Next frontier?
                - AutoML
                - New Language architecture?
                LLVM support at the compiler level


DeepDream
PoseNet


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
    MXNET (Amazon)
    CNKT (Microsoft)

Model Zoo

GANS models


Conda library installation


conda install matplotlib
conda install nb_conda
conda install scikit-image
conda install keras
conda install tensorflow
conda install tensorflow-gpu
conda install regex

conda install -c bioconda tqdm 
