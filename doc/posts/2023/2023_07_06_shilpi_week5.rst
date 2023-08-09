Design Matrix Implementation and Coding with PEP8
=================================================

What I did this Week
~~~~~~~~~~~~~~~~~~~~

This week I worked on improving the design_matrix that I'd previously created. It wasn't according to the PEP8 standards, I managed to modify it according to the appropriate format. 
I mostly spent a bunch of time implementing functions under the Fit class in CTI. In order to achieve this task I had to look into functions which had already been implemented in DKI and QTI, as CTI is an extension of DKI and has similarities with the QTI model. 
My procedure for this part was going through the 2 different modules and comparing the functions, and making note on functions which would have to be implemented in CTI as CTI has a very different design_matrix. 
Although this implementation is not completely done, I learnt a lot. 

What is coming up next Week
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This week I intend to further implement the Fit class and also generate tests for the already implemented functions under the Fit class. 
And also write tests to make sure that the signals generated in the QTI model are the same as the ones done in CTI_pred. 
I also intend on changing the order of parameters of covariance tensor as CTI has a lot of similarities with the QTI module, and in order to use QTI functions, we need to make sure that the order of parameters under covariance tensor in QTI is same as order of parameters in CTI. 
