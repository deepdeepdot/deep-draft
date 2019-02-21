### Normal distribution, trees, recursion

![Fixing Problems Image](../img/fixing_problems.webp)


### Cyber Teenagers

    Problem ->
      The department of citizen cloning has asked us to design teenagers
      so we can recreate a cyber town that looks like our current teenage population.
      Currently, the current cyber town proposal doesn't resemble at all.

      Loop
        a = give me a random height() # from a uniform(4, 6)
        count,
        sort and display -> 9 kids: of 3 different heights
      
      This is unacceptable! We need to fix this before the showtime.


### Teenager's height

    Why do we care?
    He is 6 feet 10. That's very tall!!
    He is 4 feet 10. He/she is short!?

    Relative to what? relative to the mean... average

    Height cannot be negative, but also unlikely we have 1 feet people.
    Also unlikely we have 100 feet tall people.

    -> Distribution of teenager's height?
    Normal, mean = 5 feet, std deviation +- 2 feet?


### What's normal?

    A) Let's start with the normal standard distribution
       N(0, 1), what does it mean?

       Area, 67% within 1 std dev
       Area, 95% within 2 std dev
       Area, 99% within 3 std dev

    B) What's our distribution?
       Average = (get all heights) / count
       Std dev = some ugly formula -> 

       N(mu, sigma)


### Let's make it normal

    normal?
      Use mumpy -> np.random.normal(mu, std, 2) # 100 samples
      Plot graph

      samples = 2
      samples = 10
      samples = 20
      samples = 30

      samples = 100


### Let's resize the teenagers

C) Let's fix the simulator

Loop
a = give me a random height() # from a normal(mean, stdev)
count,
sort and display

Display 9 kids of varying heights

We should get a bonus!


### Normal Distribution (Bell curve)

http://www.learningaboutelectronics.com/Articles/How-to-create-a-normal-distribution-plot-in-Python-with-numpy-and-matplotlib.php

In a normal distribution, 68% of the data set will lie within ±1 standard deviation of the mean. 95% of the data set will lie within ±2 standard deviations of the mean. And 99.7% of the data set will lie within ±3 standard deviations of the mean.

So in the following code below, we create a normal distribution with a mean centered at 90, with a standard deviation of 2, and 10000 (ten thousand) random data points created. 


#### Curve

    import numpy as np
    import matplotlib.pyplot as plt

    values= np.random.normal(90,2, 10000)
    plt.hist(values,50)
    plt.show()


https://emredjan.github.io/blog/2017/07/19/plotting-distributions/

    import scipy.stats as ss
    import numpy as npß

    def plot_normal(x_range, mu=0, sigma=1, cdf=False, **kwargs):
        '''
        Plots the normal distribution function for a given x range
        If mu and sigma are not provided, standard normal is plotted
        If cdf=True cumulative distribution is plotted
        Passes any keyword arguments to matplotlib plot function
        '''
        x = x_range
        if cdf:
            y = ss.norm.cdf(x, mu, sigma)
        else:
            y = ss.norm.pdf(x, mu, sigma)
        plt.plot(x, y, **kwargs)

    x = np.linspace(-5, 5, 5000)

    plot_normal(x, -2, 1, color='red', lw=2, ls='-', alpha=0.5)
    plot_normal(x, 2, 1.2, color='blue', lw=2, ls='-', alpha=0.5)
    plot_normal(x, 0, 0.8, color='green', lw=2, ls='-', alpha=0.5);


### mean, stdev

    FROM: https://stackoverflow.com/questions/20011494/plot-normal-distribution-with-matplotlib/20026448

    import numpy as np
    import scipy.stats as stats
    import pylab as pl

    h = sorted([186, 176, 158, 180, 186, 168, 168, 164, 178, 170, 189, 195, 172,
        187, 180, 186, 185, 168, 179, 178, 183, 179, 170, 175, 186, 159,
        161, 178, 175, 185, 175, 162, 173, 172, 177, 175, 172, 177, 180])  #sorted

    fit = stats.norm.pdf(h, np.mean(h), np.std(h))  #this is a fitting indeed
    pl.plot(h,fit,'-o')
    pl.hist(h,normed=True)      #use this to draw histogram of your data
    pl.show()                   #use may also need add this 


### more!
    import numpy as np
    import scipy.stats as stats
    import matplotlib.pyplot as plt

    h = [186, 176, 158, 180, 186, 168, 168, 164, 178, 170, 189, 195, 172,
      187, 180, 186, 185, 168, 179, 178, 183, 179, 170, 175, 186, 159,
      161, 178, 175, 185, 175, 162, 173, 172, 177, 175, 172, 177, 180]
    h.sort()
    hmean = np.mean(h)
    hstd = np.std(h)
    pdf = stats.norm.pdf(h, hmean, hstd)
    plt.plot(h, pdf) # including h here is crucial


- https://en.wikipedia.org/wiki/Normal_distribution
- https://www.mathsisfun.com/data/standard-normal-distribution.html

      import numpy as np
      import matplotlib.pyplot as plt

      mean = 0; std = 1; variance = np.square(std)
      x = np.linspace(-5,5,1000)
      f = np.exp(-np.square(x-mean)/2*variance)/(np.sqrt(2*np.pi*variance))

      plt.plot(x,f)
      plt.ylabel('gaussian distribution')
      plt.show()


#### Normal Distribution v2

    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.mlab as mlab
    import math

    mu = 0
    variance = 1
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x, mlab.normpdf(x, mu, sigma))
    plt.show()


### Kadenze class on Tensorflow

- Material verbatim from:
https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow/info

- Amazing course (if you have some math and python)


#### Celebrity Image Dataset

    import matplotlib.pyplot as plt
    import numpy as np
    import os

    files = os.listdir("celebrity_images")
    img = plt.imread(os.path.join("celebrity_images", files[50]))
    img.shape
    plt.imshow(img)
    plt.show()

    img[0,0,0] # R for (0,0) pixel
    img[0,0,1] # G for (0,0) pixel
    img[0,0,2] # B for (0,0) pixel
    plt.imshow(img[:,:,0], cmap='gray')
    plt.imshow(img[:,:,1], cmap='gray')
    plt.imshow(img[:,:,2], cmap='gray')

* Python challenge: how can we load all the images into 'imgs' using list comprehensions?


#### Celebrity Statistics

    paths = [os.path.join("celebrity_images/", f) for f in files]
    imgs = [plt.imread(path) for path in paths]

    plt.imshow(imgs[3])
    plt.imshow(imgs[2])
    data = np.array(imgs)
    data.shape

    mean_img = np.mean(data, axis=0)
    plt.imshow(mean_img.astype(np.uint8))

    std_img = np.std(data, axis=0)
    plt.imshow(std_img.astype(np.uint8))
    plt.imshow(np.mean(std_img, axis=2).astype(np.uint8))


#### Celebrity pixel histogram

    flattened = data.ravel()
    flattened.shape
    100 * 218 * 178 * 3

    plt.close('all')
    plt.hist(flattened.ravel(), 255)


#### Plot celebrity statistics

    bins = 20
    fig, axs = plt.subplots(1, 3, figsize=(12, 6), sharey=True, sharex=True)

    # Original Image Data
    axs[0].hist((data[0]).ravel(), bins)
    axs[0].set_title('img distribution')

    # Mean Data
    axs[1].hist(mean_img.ravel(), bins)
    axs[1].set_title('mean distribution')

    # Difference (Data[0] - Mean) distribution
    axs[2].hist((data[0] - mean_img).ravel(), bins)
    axs[2].set_title('(img - mean) distribution')


#### Plot Normalized

    # 1. Difference (Data[0] - Mean) distribution
    axs[0].hist((data[0] - mean_img).ravel(), bins)
    axs[0].set_title('(img - mean) distribution')

    # 2. stdev
    axs[1].hist(std_img.ravel(), bins)
    axs[1].set_title('std deviation distribution')

    # 3. normalized
    axs[2].hist(((data[0] - mean_img) / std_img).ravel(), bins)
    axs[2].set_title('((img - mean) / std_dev) distribution')

    # A normalized distribution has what shape?
    axs[2].set_xlim([-150, 150])
    axs[2].set_xlim([-50, 50])
    axs[2].set_xlim([-10, 10])
    axs[2].set_xlim([-5, 5])


### Tree and recursion


### Life of a Calculator

- Low-level: Little Man Computer?
https://en.wikipedia.org/wiki/Little_man_computer
https://hacks.mozilla.org/2017/02/a-crash-course-in-assembly/

- High-level:
  * Parser: "4 / 2 + 3 * 4" -> tree
  * Eval: tree -> number


#### Recursive defintions: Fibonacci

    Fibonacci
    fib(0) = 1
    fib(1) = 1
    fib(n) = fib(n-1) + fib(n-2) # for n > 1

    Python challenge: implement fibonacci

    Factorial
    fact(0) = 1
    fact(1) = 1
    fact(n) = n * fact(n-1) # for n > 1

* Python challenge: implement factorial and fibonacci


#### Recursive defintions: Expressions

    # Rules
    expression = [number]
    expression = [operation, expression, expression]

    num_expression = [6]

    # Expression for: "4 / 2 + 3 * 4"
    computational_expression = [
      '+',
      ['/', [4], [2]],
      ['*', [3], [4]]
    ]

* Lisp: a functional language based on lists
* https://twobithistory.org/2018/10/14/lisp.html


####  Print the expression

    def print_expression(expression):
      if (len(expression) == 1):
        print(expression[0])
      else:
        print(expression[0])
        print_expression(expression[1])
        print_expression(expression[2])

    print_expression(num_expression)
    print_expression(computational_expression)

    Python challenge: how can we eval() the expression recursively?


####  Compute the expression

    operations = {
      '+': lambda a, b: a + b,
      '-': lambda a, b: a - b,
      '*': lambda a, b: a * b,
      '/': lambda a, b: a / b
    }

    def eval_expression(expression):
      if (len(expression) == 1):
        return expression[0] # must be a number, right?
      else:
        operand = expression[0] 
        left = eval_expression(expression[1])
        right = eval_expression(expression[2])
        return operations[operand](left, right)

    print("Total: ", eval_expression(computational_expression))


#### Tree: Node class

    Class Node:
      data
      right (pointer to Node)
      left (pointer to a Node)
      isLeaf(): right and left are null

    def eval(<data>, left=a=node, right=a-node):
      return 1


### Tree: let's plant one

    def node(data=None, left=None, right=None):
      return {
        "data": data,
        "left": left,
        "right": right
      }

    root = node('+',
      left=node('/',
        left=node('4'),
        right=node('2')
      ),
      right=node('*',
        left=node('3'),
        right=node('4')
      )
    )


#### let's traverse!

    def visit(node):
      if node != None:
        visit(node["left"])
        print(node["data"])
        visit(node["right"])
      
    visit(root)
    # What if we "print" after visiting right and left?


### Node class

    # Expression for: "4 / 2 + 3 * 4"
    root = Node(operations['+'],
      left=Node(operations['/'], left=Node(4), right=Node(2)),
      right=Node(operations['*'], left=Node(3), right=Node(4))
    )
    total = root.eval(root)

    class Node:
      def __init__(self, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

      def eval(self, node):
        return []

Implement 'eval()'


#### Node class

    class Node:
      def __init__(self, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

      def eval(self, node):
        if type(node.data) == int or type(node.data) == float:
          return node.data

        value_left = self.eval(node.left)
        value_right = self.eval(node.right)
        return node.data( value_left, value_right )

TODO: check      
      def __str__(self):
        if self is null: return ""
        return __str__(self.left) + self.data + __str__(self.right)


### Exercises
1. Implement __str__ for printing the tree
2. Implement an indented version for printing the tree


### Parser?

  - What's wrong with our calculator?
    - It's hardcoded for the expression: "4 / 2 + 3 * 4"
    - No support for parenthensis, unary operators (negative)
    - How can we make it dynamic? How do we support variables?

  - Yacc and Lex for Python
    - https://www.dabeaz.com/ply/
    - https://www.dabeaz.com/ply/ply.html
    - https://github.com/dabeaz/ply

  - Calculator: input = math expression
  - Interpreter: input = computer program source code
      - operations: for-loop, if, switch, expressions
  - Reference
    - http://openbookproject.net/thinkcs/python/english3e/trees.html
    - https://pypi.org/project/binarytree/


#### Tensorflow is a computational graph
    # $ conda activate nanos
    # $ conda install tensorflow-gpu # or tensorflow
    import tensorflow as tf

    sess = tf.Session() # Create a session

    # z = "4 / 2 + 3 * 4"
    z = tf.add(tf.divide(4.0, 2.0), tf.multiply(3.0, 4.0))
    z = tf.add(tf.divide(tf.Constant(4.0), tf.Constant(2.0)),
                tf.multiply(tf.Constant(3.0), tf.Constant(4.0)))
    computed_z = sess.run(z)

    y = tf.linspace(-3.0, 3.0, 100)
    computed_y = sess.run(y)
    computed_z2 = sess.run(z)

    sess.close() # Close the session

    computed_z
    computed_z2
    computed_y


#### Tensorflow Variables
TODO: Test this
    # https://www.tensorflow.org/api_docs/python/tf/Variable

    import tensorflow as tf
    A = tf.Variable([[1,2,3], [4,5,6], [7,8,9]], dtype=tf.float32)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      print(sess.run(A[:2, :2]))  # => [[1,2], [4,5]]

      op = A[:2,:2].assign(22. * tf.ones((2, 2)))
      print(sess.run(op))  # => [[22, 22, 3], [22, 22, 6], [7,8,9]]

    w = tf.Variable(4.0, name="w")

    with tf.Session() as sess:
      # Run the variable initializer.
      sess.run(w.initializer)
      
      z = tf.add(tf.divide(4.0, 2.0), tf.multiply(3.0, w))
      computed_z = sess.run(z)

      w.assign(10)
      computed_z2 = sess.run(z)


#### Tensorflow challenges
  1. Compute the Bell Curve using tensorflow and display the image
  2. How about using the tf.placeholder() instead of harcoding constant values?


