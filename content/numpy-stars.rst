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

.. highlight:: python

Exploring our local stellar neighbourhood
-----------------------------------------

The Gaia sattelite has imaged around 1.8 billion stars and other objects.
Among the things it measured was the "parallax" of each object, which is how much a star seems to "move" in relationship to other stars as the sattelite traveled around the sun and observed it from different angles.
It's the same thing as how nearby objects appear to be in different positions when you look at them with your left versus your right eye.
We can use the amount of parallax to determine how far away a star is.

.. image:: img/numpy/parsec.png
 
What is an array?
-----------------

