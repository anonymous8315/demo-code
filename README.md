# demo-code

Repo with code for submission under review.

To run protein design with Gibbs sampling using Progen2 model likelihoods, first clone the [Progen repo](https://github.com/salesforce/progen), then run the command 

`python mcmc_sampling.py`

You can specify the starting sequence with the `context` flag, for example with 

`python mcmc_sampling.py --context 1MSETDGEAEETGQTHECRRCGREQGLVGKYDIWLCRQCFREIARSMGFKKYS2`.
