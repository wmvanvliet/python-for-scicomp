.. _numpy_stars:

NumPy Stars
===========

.. questions::

   - Why use NumPy instead of pure python?
   - How to use basic NumPy?
   - What is vectorization?

.. objectives::

   - Understand the Numpy array object
   - Be able to use basic NumPy functionality
   - Understand enough of NumPy to search for answers to the rest of your questions ;)

   We expect most people to be able to do all the basic exercises here.
   For those who are already familiar with the basics and desire something more challenging, we have more advanced exercises at the end.

.. admonition:: Example context
   :class: demo

   Each lesson is placed inside an example scientific setting.
   In this lesson, we will compute the distance of the stars in our night sky, using the data collected by the Gaia sattelite.

NumPy is the most used library for scientific computing.
Even if you are not using it directly, chances are high that some library uses it in the background.
It helps you work with "data", as in large amounts of numbers, by providing:

1. A high-performance multidimensional array object to store data in computer memory
2. Efficient routines to manipulate data, such as reading and writing, splitting and concatenation, and other transformations
3. Efficient routines to perform computations on data, such as statistics and linear algebra

Let's explore the fundamentals of this foundational library by analyzing some astronomical data.

.. highlight:: python

Part 1: Exploring our local stellar neighbourhood
-------------------------------------------------

The `Gaia sattelite <https://www.esa.int/Science_Exploration/Space_Science/Gaia>`__ has imaged around 1.8 billion stars and other objects.
Among the things it measured was the "parallax" of each object, which is how much it seems to "move" in relationship to very distant stars as the sattelite traveled around the sun and observed it from different angles.
We can use the amount of parallax to determine how far away a star is.

.. figure:: img/numpy/parsec.png
   :width: 300px
   :align: center

   Visual explanation of the parallax effect. Image `retrieved from Wikipedia <https://commons.wikimedia.org/wiki/File:Stellarparallax_parsec1.svg>`__. Public domain.

A sample of the Gaia data can be found in :download:`../resources/data/numpy/local_stars.csv`.
This is a *comma separated value (CSV)* file, where each line contains 6 numbers describing a single object.
The first line of that file is the "header" indicating the meaning of each number::

    with open("../resources/data/numpy/local_stars.csv") as f:
        # The file handle `f` acts as a generator that yields lines of text.
        first_line = next(f)  # take a single item from the generator
        names = first_line.split(",")  # split on commas to get a list of individual names
    print(names)

We can use one of NumPy's functions to load the entire file into memory::

    import numpy as np  # we usually use `np` as shorthand
    # We set `skip_header=1` to skip the first line containing the description of the values.
    data = np.genfromtxt("../resources/data/numpy/local_stars.csv", delimiter=",", skip_header=1)
    print(data)

The ``data`` variable is a NumPy array::

   print(type(data))
 

What is an array?
-----------------

An array is a "grid" of values, with all the same type, tightly packed into memory side-by-side.
This is why we skipped the first line of the file containing ``str`` type descriptions, so that we were only reading numbers.
In our case, the data type of the array is::

    print(data.dtype)

Arrays can have any number of dimensions::

    print(data.ndim)

You can think of a 1D array as a *list* and a 2D array as a *table*.
In mathematics, we call a 1D array a *vector*, a 2D array a *matrix*, and an array with 3 or even more dimensions a *tensor*.
An array can even have zero dimensions if it represents a single *scalar* number.
Regardless of how many dimensions the array has, it is always a Python object of type :class:`numpy.ndarray`.

Let's see how many stars we have::

    print(data.shape)

We have information on more than a million stars, and for each star we have 6 values, corresponding to (see ``names`` above):

0. ``right_ascension``: horizontal position of the star along the celestial equator (degrees)
#. ``declination``: vertical position of the star above/below celestial equator (degrees)
#. ``parallax``: amount the star's position in the sky changes throughout the year
#. ``magnitude_g``: how bright the star appears in our sky (inverse Logarithmic scale)
#. ``magnitude_red``: red-component of the color of the star
#. ``magnitude_blue``: blue-component of the color of the star


Working with arrays
-------------------

The way the array is laid out in memory makes it very fast to select portions of the data, by "indexing" or "slicing" the array.
You can index/slice arrays in a similar manner as Python lists, but you specify an index/slice for each dimension.
At the moment, we are only interested in the parallax value of each star, so let's extract that column and store it in a variable of its own::

    parallax = data[:, 2]  # `:` is a shorthand for "select everything along this dimension"
    print(parallax)

Now for some basic math.
The parallax values are currently in `milli-arc-seconds <https://en.wikipedia.org/wiki/Minute_and_second_of_arc>`__.
Let's convert that into distance.
A common distance metric is the `"parsec" <https://en.wikipedia.org/wiki/Parsec>`__, which is the distance at which an object has a parallax of exactly one arc-second, which is 3.26156 light-years::

    distance_parsecs = 1 / (parallax / 1000)  # 1000 milli-arc-seconds in an arc-second
    distance_ly = distance_parsecs * 3.26156

Note that we didn't write any ``for``-loops.
In NumPy, arithmetic operations on arrays are applied to every element in that array.
It is very fast, because it is implemented in C or Fortran and makes use of specialized CPU instructions that perform operations on many values at once.
We say an operation is "vectorized" when the looping over elements is carried out by NumPy internally.
Common mathematical operations include: ``+`` (:data:`numpy.add`), ``-`` (:data:`numpy.subtract`), ``*`` (:data:`numpy.multiply`), ``/`` (:data:`numpy.divide`), ``.T`` (:func:`numpy.transpose`), :data:`numpy.sqrt`, :func:`numpy.sum`, :func:`numpy.mean`, ...

.. note::
   In NumPy, ``*`` performs element-wise multiplication and ``@`` performs matrix multiplication. This is different from other languages such as MATLAB or Julia.

Let's examine the closest star in the dataset::

    distance_to_closest_star = distance_ly.min()  # returns the minimum value
    index_of_closest_star = distance_ly.argmin()  # returns the index of the minimum value
    print(distance_to_closest_star, index_of_closest_star)

The closest star is `Proxima Centauri <https://en.wikipedia.org/wiki/Proxima_Centauri>`__ at 4.2465 light-years.
To see all the information we have on it, we can select the entire row using the index we just obtained::

    print(names)  # the column names for reference
    print(data[index_of_closest_star, :])  # select the row corresponding to Proxima Centauri

NumPy has many other ways to select data.
For example, let's see if we can show some information on the 10 stars closest to us.
To do this, we must first order the stars by distance::

    order_by_distance = np.argsort(distance_parsecs)  # argsort returns indices, not values
    closest_indices = order_by_distance[:10]  # select the first 10 elements
    print(closest_indices)

We now have a list (technically a 1D numpy.array) of integer indices of the stars closest to us.
We can select multiple rows by indexing the array with such a list::

    closest_stars = data[closest_indices, :] 
    print(distance_ly[closest_indices])

Can we find the `Big Dipper <https://en.wikipedia.org/wiki/Ursa_Major>`__?

.. figure:: img/numpy/Big_Dipper.jpg
   :width: 300px
   :align: center

   The Big Dipper as seen from Fujian. Image `retrieved from Wikipedia <https://en.wikipedia.org/wiki/Big_Dipper#/media/File:Big_Dipper_20210116.jpg>`__. CC-BY


It lies within a section of the sky with a `right ascension <https://en.wikipedia.org/wiki/Right_ascension>`__ from 160 to 210 degrees, and a `declination <https://en.wikipedia.org/wiki/Declination>`__ from 45 to 65 degrees::

    right_ascension = data[:, 0]
    declination = data[:, 1]

To select elements from an array based on some condition, we can create an index based on boolean values.
Recall that in Python, comparison operations (``==``, ``<``, ``<=``, ``>``, ``>=``) produce a boolean (``True``/``False``) value.
See what happens when you apply them to an array::

    print(right_ascension > 160)

It produces a new array, of the same dimensions as the original array, but filled with ``True``/``False`` values indicating for each element whether the comparison holds true or not.
We call this a "boolean mask".
We can combine multiple masks using Python's boolean "and" operator ``&``::

    boolean_mask = (right_ascension > 160) & (right_ascension < 210) & (declination > 45) & (declination < 65)

Once you have a boolean mask, you can apply it to any array as long at the dimension along which you are selecting has the same number of elements as the mask::

    sky_section = data[boolean_mask, :]
    print(sky_section.shape)

In our dataset, there are more than 15k stars in our chosen section of sky.
The Big Dipper should consist of the 7 brightest ones.
Column 3, named `magnitude_g` denotes bright the star looks as seen by the Gaia sattelite.
The `magnitude scale <https://en.wikipedia.org/wiki/Magnitude_(astronomy)>`__ works "in reverse", such that very bright stars have a magnitude of 1 and weaker stars have a larger magnitude.

.. code::

    # Select the 7 brightest stars in our section of sky.
    order_by_magnitude = np.argsort(sky_section[:, 3])
    sky_section = sky_section[order_by_magnitude, :]
    big_dipper = sky_section[:7]

    # Make a figure with the position of the stars in the sky. You can take this at face
    # value for now. We will cover how to make figures in a later lesson.
    import matplotlib.pyplt as plt
    fig, ax = plt.subplots()
    ax.scatter(big_dipper[:, 0], big_dipper[:, 1])
    ax.xaxis.set_inverted(True)  # right ascension goes from right-to-left
    ax.set_aspect("equal")
    ax.set_xlabel("right ascension (degrees)")
    ax.set_ylabel("declination (degrees)")


Clever and efficient use of these operations is a key to NumPy's speed: you should try to cleverly use these selectors (written in C) to extract data to be used with other NumPy functions written in C or Fortran.
This will give you the benefits of Python with most of the speed of C.

.. seealso::

   :ref:`Numpy basic indexing docs <basics.indexing>`


Exercises 1
-----------

.. challenge:: Exercises: Numpy-1

   Proxima Centauri is part of a 3-star system called `Alpha Centauri <https://en.wikipedia.org/wiki/Alpha_Centauri>`__.
   The other two stars are very close together and appear as a single object (Alpha Centauri AB) in this dataset.
   
   #. What is the row index of the binary star Alpha Centauri AB?
      It is the second closest star to us.
      What is the distance from us to Alpha Centauri AB?
   #. What is the distance from us to the brightest star in the dataset (`Sirius <https://en.wikipedia.org/wiki/Sirius>`__)?
   #. How many stars are there in the dataset that are even brighter than the brightest star in the Big Dipper (`Alioth <https://en.wikipedia.org/wiki/Alioth>`__)?



Part 2: Estimating the distance of really far away stars
--------------------------------------------------------

When stars are very far away, the parallax is so small even the Gaia satellite cannot reliably measure it.
Still, we can estimate roughly how far away they are based on a striking relationship between the brightness and the color of a star.
A `Hertzsprung-Russel diagram <https://en.wikipedia.org/wiki/Hertzsprung%E2%80%93Russell_diagram>`__ illustrates this.
Let's make one!

In our data, column 3 contains the measurements of the apparent brightness of each star according to Gaia's sensors.
For the Hertzsprung-Russel diagram, we need to estimate the intrinsic brightness (the luminosity) of the stars, compensating for the fact that the further a star is, the dimmer it appears to us.
We will use the formula given in the `official Gaia paper <https://doi.org/10.1051/0004-6361/201832843>`__ for this::

    magnitude_g = data[:, 3]  # apparent magnitude
    magnitude = magnitude_g + 5 * np.log10(parallax) - 10  # true magnitude

In our diagram, we want to show the color of a star on a scale from "very blue" to "very red", with yellow stars in between.
We can compute this by taking the difference between the blue and red components of its color::

    magnitude_red = data[:, 4]
    magnitude_blue = data[:, 5]
    color = magnitude_blue - magnitude_red

Now we have everything we need to make the plot. Again, just take this plotting code at face value for now::

    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 8))
    plt.scatter(
        color, magnitude, s=1, c=distance_ly, cmap="plasma", alpha=0.1
    )
    cb = plt.colorbar(label="Distance (light years)", fraction=0.05, aspect=40)
    cb.solids.set(alpha=1)  # make the colors in the colorbar not be transparent
    plt.gca().invert_yaxis()  # put bright stars on top, dim stars at the bottom
    plt.title("Hertzsprung-Russell diagram")
    plt.xlabel("Color (Blue ⇨ Red)")
    plt.ylabel("Magnitude")
    plt.tight_layout()

Other than the main sequence stars, you can see the red giants branching off at the top, bright but red, and the white dwarfs at the bottom, very dim but white.

Based on this chart, we can derive a formula to translate the color of a main sequence star (so not a red giant or white dwarf) to its true brightness.
Then, we can compare that true brightness with its apparent brightness to deduce how much dimmer it appears to us than it actually is, hence how far away it must be.


Creating arrays
---------------

The first step is to extract a bunch of ``(color, magnitude)`` pairs along the main sequence, free from all the outliers.
The strategy is to divide the color spectrum into 100 bins and take the median magnitude in each bin.
With the vast majority of the stars being main sequence stars, the median should be very representative of the general curve of the main sequence.

There are various ways to create an array with equally spaced numbers.
For example, there is the NumPy equivalent of Python's :func:`range` function, :func:`numpy.arange`::

    bins = np.arange(0.5, 4, step=0.04)

But for our purposes, :func:`numpy.linspace` is even better::

    n_bins = 100
    bins = np.linspace(0.5, 4, num=n_bins)  # 100 numbers evenly spread between 0.5 and 4

We could assign stars to their respective color bins using boolean masking, but NumPy offers the :func:`numpy.digitize` function especially for this purpose::

    bin_assignment = np.digitize(color, bins)
    print(bin_assignment)

The function gives for each element of the ``color`` array (as an integer index), the index of the bin it belongs to.
Since each element of the ``color`` array is the color of a star, we have effectively assigned each star to a bin.
Now, we can compute for each bin, the median brightness and color of all the stars in that bin.
We will do this using a ``for`` loop and use a typical pattern for creating NumPy arrays: we first collect the values into a Python list and then convert that list into a NumPy array::

    # We collect the median colors and magnitudes in these lists.
    bin_colors = list()
    bin_magnitudes = list()

    # Loop over all the bins and compute the median color/magnitude.
    for bin_index in np.arange(n_bins):
        bin_color = np.median(color[bin_assignment == bin_index])
        bin_magnitude = np.median(magnitude[bin_assignment == bin_index])
        bin_colors.append(bin_color)
        bin_magnitudes.append(bin_magnitude)

    # Convert the Python lists to NumPy arrays.
    bin_colors = np.array(bin_colors)
    bin_magnitudes = np.array(bin_magnitudes)

.. note::
   Appending values to a Python list is very fast, but appending values to a NumPy array is very slow. This is why we first collect the values in a Python list and then convert it to a NumPy array.

.. seealso::

   `Numpy array creation docs <https://numpy.org/doc/stable/user/basics.creation.html>`_


Doing linear algebra
--------------------

If we fit a curve through the  ``(color, magnitude)`` combinations, we obtain a function to predict a star's magnitude based on its color.
We can do this with a bit of linear algebra.
To get a feel for how the curve should look, let's draw the ``(color, magnitude)`` combinations on top of the Hertzsprung-Russel diagram (assuming it is still open, re-create it if it isn't)::

    plt.plot(bin_colors, bin_magnitudes, color="green", linewidth=2)

The curve is not a straight line, but something like a fifth order polynomial should be pretty good fit:

.. math::
   :name: A fifth order polynonial function

   \text{magnitude} = \beta_5\,\text{color}^5 + \beta_4\,\text{color}^4 + \beta_2\,\text{color}^3 + \beta_2\,\text{color}^2 + \beta_1\,\text{color} + \beta_0

"Fitting the curve" now means choosing the optimal :math:`\beta` values so that when we input the ``color``, we get a good prediction of ``magnitude``.
A typical way to do this is a machine learning technique called `ordinary least squares (OLS) <https://en.wikipedia.org/wiki/Ordinary_least_squares>`__, where we collect everything we know in a matrix ``X`` and everything we wish to predict in a matrix ``Y`` and compute:

.. math::
   :name: The ordinary least squares function

   \beta = (X^T X)^{-1} X^T Y

The resulting ``beta`` vector contains the optimal linear combination (in the least-squares sense) of the columns of ``X`` to predict the columns of ``Y``.
So, we create a 2D array ``X`` where the columns are ``bin_colors`` raised to different powers and set ``Y`` to ``bin_magnitude``.
The code below does this, while demonstrating how to create a 2D array by gluing 1D arrays together, and how to generate an array containing only ``1``'s::

    X = np.column_stack((
        np.ones_like(bin_colors),  # bin_colors ** 0
        bin_colors,                # bin_colors ** 1
        bin_colors ** 2,
        bin_colors ** 3,
        bin_colors ** 4,
        bin_colors ** 5,
    ))
    Y = bin_magnitudes

To compute the OLS formula, we need a bunch of linear algebra functionality:

* matrix multiplication, which is written as ``@`` in Python
* matrix transpose, which NumPy arrays support through their property ``.T``
* matrix inversion, which is done through :func:`numpy.linalg.inv`

Using NumPy, the Python code can look very much like the math formula::

    beta = np.linalg.inv(X.T @ X) @ X.T @ Y
    print(beta)


Creating and documenting functions in the scientific computing style
--------------------------------------------------------------------

Now that we have ``beta``, we can use formula (1) to predict a star's magnitude given its color.
Let's wrap the formula in a function, so we can give it a name and write documentation on how to use it.
NumPy comes with a style guide for writing docstrings for functions called `NumPyDoc <https://numpydoc.readthedocs.io/en/latest/format.html>`__::

    def predict_magnitude(color, beta):
        """Predict a main sequence star's inherent magnitude given its color.

        Parameters
        ----------
        color : float | array of float, shape (n_stars,)
            The color of the star(s), computed as red - blue.
        beta: array of float, shape (6,)
            The regression weights.

        Returns
        -------
        magnitude : float
            The estimated intrinsic magnitude of the star(s).
        """
        return (
            beta[5] * color ** 5
            beta[4] * color ** 4 +
            beta[3] * color ** 3 +
            beta[2] * color ** 2 +
            beta[1] * color +
            beta[0] +
        )

Let's use our ``predict_magnitude`` function to create a function that predicts the distance of a star given its color and apparent magnitude.::

    def predict_distance(magnitude_g, magnitude_red, magnitude_blue, beta):
        """Predict a main sequence star's distance given its apparent magnitude and color.

        Parameters
        ----------
        magintude_g : float | array of float, shape (n_stars,)
            The apparent magnitude of the star(s).
        magnitude_red : float | array of float, shape (n_stars,)
            The red component of the color of the star(s).
        magnitude_blue : float | array of float, shape (n_stars,)
            The blue component of the color of the star(s).
        beta: array of float, shape (6,)
            The regression weights.

        Returns
        -------
        distance : float
            The estimated distance in light-years.
        """
    color = magnitude_red - magnitude_blue
    predicted_magnitude = predict_magnitude(color, beta)
    # Inverse of the formula we used to derive intrinsic magnitude from apparent magnitude.
    predicted_distance_parsecs = np.pow(10, ((magnitude_g - predicted_magnitude) + 10) / 5)
    return predicted_distance_parsecs * 3.26156

A sample of some far away main sequence stars in the Gaia dataset can be found in :download:`../resources/data/numpy/far_stars.csv`.
Their parallax values may be unreliable, so let's predict their distance using main sequence fitting::

    far_stars = np.genfromtxt("../resources/data/numpy/far_stars.csv", delimiter=",", skip_header=1)
    print(far_stars.shape)

    far_stars_distance = predict_distance(far_stars[:, 3], far_stars[:, 4], far_stars[:, 5], beta)
    print(far_stars_distance)
