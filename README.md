# Sequential Hidden Variable Model

TODO:

* Allow models not to condition on actions.
* Allow models to condition on invariant information throughtout a sequence.
* Allow models to have independent transition or observation.
* Allow algorithms to handle non-reparameterized models and the reparameterized ones.
* Make all models and proposals to share the same state dict
* Make all models and proposals  be able to condition on some prior knowledges.
* Modify algorithms to accomodate the above changes
* Modify SMC to be TwistedSMC and returns only loss term
* Implement FilteringProposal and SmoothingProposal respectively, and replace the current ones
* Make twisters a separate class.
* Make sure the shape of observations and actions are expand to batch_shape before passing to models and proposals, where the batch_shape includes the particle dimension when there are multiple particles.
* Make sequence elbo suitable for the general case
