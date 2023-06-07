Signal Creation & Paper Research: Week2 Discoveries
===================================================

.. post:: June 05 2023
    :author: Shilpi Prasad
    :tags: google
    :category: gsoc



What I did this week
~~~~~~~~~~~~~~~~~~~~
I worked through this research paper, and found some relevant facts to the tasks at hand, such as the different sources of kurtosis. One other important fact I found out was that DDE comprises 2 diffusion encoding modules characterized by different q-vectors (q1 and q2 ) and diffusion times. This fact is important because, CTI approach is based on DDE's cumulant expansion, and the signal is expressed in terms of 5 unique second and fourth-order tensors. I also found out about how the synthetic signals could be created using 2 different scenarios, which comprises a mix of Gaussian components and a mix of Gaussian and/or restricted compartments. 
The major time I spent this week was in creating synthetic signals, and therefore in creating simulations.


What Is coming up next week
~~~~~~~~~~~~~~~~~~~~~~~~~~~
I intend on finishing the simulations with appropriate documentation and theory lines. If time permits, I'll resume working on the cti.py file and its tests section.


Did I get stuck anywhere
~~~~~~~~~~~~~~~~~~~~~~~~
I didn't get stuck, however it did take me a while to go through all the code that I could possibly be needing in my simulations, and also in understanding the theory behind those codes.
