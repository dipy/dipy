Re-Engineering Simulation Codes with the QTI Model and Design Matrix
====================================================================

.. post:: Jun 19 2023
    :author: Shilpi Prasad
    :tags: google
    :category: gsoc



What I did this week
~~~~~~~~~~~~~~~~~~~~

I had to change the ``cti_test.py`` file as the signals generated were not exactly correct. I was advised to follow the multiple gaussian signal generation method. While doing this I had to look closely at several already implemented methods and go in depth to understand how those functions were achieving the desired output.
The multiple gaussian signal generation method is preferred because the CTI signal generation closely resembles the multiple gaussian signals. We're using the multiple gaussian signals so that we can have a priori of what to expect from the outcome, if we fit our model to this signal.
I also managed to implement the design matrix for the CTI tensor and managed to save it up in the ``utils.py`` file. The design matrix is a crucial component of the CTI tensor as it represents the relationships between the different variables in our model. By accurately modeling these relationships, we can generate more realistic simulations and gain a deeper understanding of the CTI tensor.
The link of my work: :ref:`Here<https://github.com/dipy/dipy/pull/2816>`



What is coming up next Week
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This week I'll work on fitting CTI on multiple Gaussian simulations and see if it produces the expected output. And therefore, work on improving it. This may require implementing a bunch of methods for the Fit class.

Did I get stuck anywhere
~~~~~~~~~~~~~~~~~~~~~~~~

No, I didn't get stuck anywhere.
