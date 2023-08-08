Creating signal_predict Method: Testing Signal Generation
=========================================================

.. post:: July 06 2023
    :author: Shilpi Prasad
    :tags: google
    :category: gsoc


What I did this week
~~~~~~~~~~~~~~~~~~~~

This week, I worked together with my mentor to come up with a new way of arranging the elements of the design matrix. So, first I rearranged all the parameters of the covariance parameters so that they'd match with the ones in QTI. So now, the order is: the diffusion tensor, the covariance tensor, and then the kurtosis tensors. But then we decided that it would be better to put the kurtosis tensors first because then we wouldn't have to re-implement all the kurtosis methods again. So, I changed the order of kurtosis and the covariance tensors.

Also, in order to maintain the coding standards of the previously implemented models, we decided that the diffusion tensor should be divided into evals and evecs.

Therefore, because of all these changes I had to re-implement a lot of already implemented functions in CTI which also required changing the description of those functions and not only the code.

But my major time went towards writing tests for these modified codes. While writing codes, I realized that a lot of the functions needed to be modified a bit. Also, I had to import several new libraries in order for the functions to work. 

What Is coming up next week
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The testing part for the implemented method is not yet done correctly, as the signals don't yet match the expected output. So, I intend on re-implementing them by taking into consideration the suggestions provided by my mentor. This would require modifying code of some already implemented functions as well as re-writing the tests, particularly the generation of the eigenvalues and the eigen vectors. 


Did I get stuck anywhere
~~~~~~~~~~~~~~~~~~~~~~~~

I didn't exactly get stuck, but implementing the tests requires you to make sure that the shape of the tensors you're passing into a function is correct and is as expected. This took me a while to figure out. 

