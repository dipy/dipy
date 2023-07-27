Generating Fit Functions : Week 9
=================================

.. post:: July 27, 2023
     :author: Shilpi Prasad
     :tags: google
     :category: gsoc


What I did this week
~~~~~~~~~~~~~~~~~~~~

So, among several other things, this week I first started on figuring out how to run spyder on ubuntu as initially I couldn't make it run due to technical problems, but then there was this need to make sure that I'm able to edit my code while keeping the pep8 standards and the automatic formatting of my code wasn't working. So, after having done this, I made some changes in the utils.py file to make the design_matrix more readable. I also made changes in the B[:,3] and B[:, 4] which are the diffusion tensor elements as there was a typo in the sign because we realized that the diffusion tensor part should have a negative contribution as it represents a signal decay

One of the other tasks that I implemented was mapping all the covariance parameters from paper to its actual code implementation which created some confusion, as the conversion shown in the paper didn't quite match its implementation. This was what initially created the need to talk to the authors of the original paper. 

The other more important work I did was trying to figure out the matching of the ground truth signal values in case of anisotropic and combined DTDs. This is because the isotropic DTD signals that were being generated matched exactly the QTI signals, as in case of isotropic we've 6 non zero elements, and the rest are 0s. However in anisotropic case we had more non-zero covariance parameters (9 non-zero), similarly as in the case of combined DTD. So we figured out that the non-zero elements are being multiplied to some value which isn't correct and that this needs modifying the ccti conversion.
So, I worked on reading more about voigt notation, as the QTI parameters were implemented using that notation.
Then we looked again into the QTI paper, and felt the need to contact their author and code implementer and realized that the coding was done while keeping in mind the voigt notation conversion as well as some other factors. At the end of this we figured out the correct conversion of the ccti parameters. We noticed that some factors needed the (root2) division, while some others needed (2). Therefore, we were able to successfully figure out the correct factors that needed to be multiplied/ divided to each of the covariance parameters. 
And hence, now the signal values of all the DTDs match as expected. 
Then the other major ongoing task this week has been the implementation of the Fit class in CTI. This required me to implement some functions which might've been implemented in DKI/ QTI. This is an ongoing task and would require more work. 

Things coming up next week 
~~~~~~~~~~~~~~~~~~~~~~~~~~

After the matching of the correct signals which matched the ground truth values, we realized that a DTD with more non-zero covariance parameters might make the ccti conversion more robust while taking all cases into consideration. So, we created a DTD with mevals, its angles and the fractions. However the signals didn't match exactly. But rather than being stuck on this case, we decided to move forward for the time being. So, I'll work on making sure that all the ground truth signals match. 
But the more important work I will be doing this coming week would be to implement all the required functions in the ccti module such as the different sources of kurtosis which hasn't been implemented before as this is one of the differentiating factors of CTI. And then hopefully move on to generating the tests for these functions. 

Did i get stuck this week
~~~~~~~~~~~~~~~~~~~~~~~~~

Not at all. Some things were kind of vexing, but I didn't get stuck as there was always something else that could be done.

