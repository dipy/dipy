Modifying Test Signal Generation 
================================

.. post:: July 12 2023
    :author: Shilpi Prasad
    :tags: google
    :category: gsoc


What I did this week
~~~~~~~~~~~~~~~~~~~~

One of the tasks I did this week was modify the ``cti_design_matrix`` again, as asked by my mentor to make the code more readable. The initial code was following pep8 standard but it wasn't very easy to read, but now it is.
Also, I realized that the main reason my signals weren't matching the ground truth values before at all was because the eigenvalues and eigenvectors of the diffusion tensor distribution were wrong. This was because, before I tried getting D_flat by doing: ``np.squeeze(from_3x3_to_6x1(D))`` which returned a tensor of shape ( 6, ). But in this case, it returned the diffusion tensor elements in the order : Dxx, Dyy, Dzz  and so on which isn't the correct format of input expected for the "from_lower_triangular" function. So, initially, we were doing : ``evals, evecs = decompose_tensor(from_lower_triangular(D_flat))`` where the from_lower_triangular function is returning a tensor of shape: (3,3). But then I realized that rather than calculating D_flat, we can simply do: ``evals, evecs = decompose_tensor(D_flat)``. Following this approach gave the correct value of "evals and evecs". So, now we have the correct values of "evals and evecs" which made the signals come closer to the ground truth signals, but we still don't have the signals completely matching the ground truth signals.
Another problem we realized was that while passing "C", covariance tensor parameters, we needed to make sure that we were passing the modified C parameters, that is "ccti". This again helped in bringing the signals to the expected values.
So, after talking things through with my mentor, and analyzing the QTI paper, we came to a few conclusions which could be done to improve the signal values. 

What is coming up next week 
~~~~~~~~~~~~~~~~~~~~~~~~~~~

We realized that there might be a slight probability that there is a typo somewhere in the actual implementation of the QTI signals. So we decided to contact the original author and code implementer of QTI. 
Also, one important thing I intend on doing is check the papers, and see the conversions are being done correctly, that is we are taking into consideration the (root2) factor which is present for some covariance tensor elements in the original paper. This is because, for the isotropic case we observe that the signals are matching perfectly because in the isotropic case all the (root2) parameters of the original covariance elements are zero. 
Another thing that I intend on doing is to create a new test method which will have some similarities to the test method in dki_tests.

Did I get stuck anywhere
~~~~~~~~~~~~~~~~~~~~~~~~

I didn't get stuck anywhere, but trying to figure out the problem with the current signal generation did take some time and required looking into the research papers. 

