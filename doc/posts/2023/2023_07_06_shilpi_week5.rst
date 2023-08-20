Design Matrix Implementation and Coding with PEP8: Week 5
=========================================================

.. post:: July 06 2023
    :author: Shilpi Prasad
    :tags: google
    :category: gsoc

What I did this Week
~~~~~~~~~~~~~~~~~~~~

This week, my work focused on two main areas: improving the design matrix and implementing methods under the Fit class in CTI.
For the design matrix improvement, I noticed that the design matrix I had previously created was not according to PEP8 standards. After some effort, I managed to modify it to comply with the appropriate format.
This week, my time was mostly consumed by implementing methods under the Fit class in CTI. As CTI is an extension of DKI and shares similarities with the QTI model, I had to look into methods already implemented in DKI and QTI. My approach involved going through these two different modules, comparing the methods, and making notes on which ones would need to be implemented in CTI. This was challenging, as CTI's design matrix is significantly different.
Although this implementation is not completely done, I was able to learn a lot. 

What is coming up next Week
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This week I intend to further implement the Fit class and also generate tests for the already implemented methods under the Fit class. 
And also write tests to make sure that the signals generated in the QTI model are the same as the ones done in ``CTI_pred``. 
I also intend on changing the order of parameters of covariance tensor as CTI has a lot of similarities with the QTI module, and in order to use QTI methods, we need to make sure that the order of parameters under covariance tensor in QTI is same as order of parameters in CTI. 
