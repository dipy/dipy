.. _gqi:

==============================
Generalised Q-Sampling Imaging
==============================

These notes are to help the user of the DIPY module understand
Frank Yeh's Generalised Q-Sampling Imaging (GQI) [reference?].

The starting point is the classical formulation of joint k-space
and q-space imaging (Calaghan 8.3.1 p. 438) using the narrow
pulse gradient spin echo (PGSE) sequence of Tanner and Stejskal:

.. math::

   S(\mathbf{k},\mathbf{q}) = \int \rho(\mathbf{r}) \exp [j 2 \pi
   \mathbf{k} \cdot \mathbf{r}] \int P_{\Delta}
   (\mathbf{r}|\mathbf{r}',\Delta) \exp [j 2 \pi \mathbf{q} \cdot
   (\mathbf{r}-\mathbf{r'})] \operatorname{d}\mathbf{r}'
   \operatorname{d}\mathbf{r}.

Here $S$ is the (complex) RF signal measured at spatial wave number $\mathbf{k}$
and magnetic gradient wave number $\mathbf{q}$.

$\rho$ is the local spin density (number of protons per unit volume
contributing to the RF signal).

$\Delta$ is the diffusion time scale of the sequence.

$P_{\Delta}$ is the averages diffusion propagator (transition
probability distribution).




