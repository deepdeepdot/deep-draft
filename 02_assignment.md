Given

    X = np.linspace(-5,5,100)
    Y = lambda x: x**2 - 3*x + 10

    plt.plot(X, Y(X))

Implement an algorithm that finds the minimum of the function Y.
The minimum (x, y) has the property that the slope is zero.

Given two points p1, p2, the slope is defined as:

slope m = (p2.y - p1.y) / (p2.x - p1.x)

    minimum, steps = find_minimum(domain=X, function=Y)

    where minimum is of the shape (x, y)
    steps: number of iterations to find this minimum

Algorithm:

    Initialize
    Pick any random point as the start
    Let's pick a point to the left and compute

    Let's pick 2 random points
    p1, p2

    p2 = random_point()

    Loop
        Compute the slope
        Is the slope equal to zero (or close to zero?)
            break
        Is the slope positive? go to the left
        Is the slope negative? go to the right


This is a contest, let's see who comes up with the "fastest" algorithm that comes with the correct result.

The priority is to get the result "right" before attempting to get the "fastest" solution. Also make sure not to get into an infinite loop.


Just to "see" that you are getting closer to the result,
keep printing the error and see that it's getting smaller.

Extra: plot the slopes against the Y
