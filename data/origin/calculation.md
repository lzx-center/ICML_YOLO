Thank you very much for the additional comments.
We realize that our previous response did not fully clarify your concerns. We wish to clarify four facts that we believe will resolve the remaining issues.

---

### The guarantee of Randomized Smoothing [1] is **probabilistic**, not absolute  

Randomized smoothing certifies robustness through Monte-Carlo sampling and therefore its guarantee is statistical. Proposition 2 on page 6 of [1] states  

> With probability at least $1 − \alpha$ over the randomness in CERTIFY, if CERTIFY returns a class $\hat{c}_A$ and a radius $R$ (i.e. does not abstain), then $g$ predicts $\hat{c}_A$ within radius $R$ around $x$: $g(x + \delta) = \hat{c}_A \forall \|\delta\|_2 < R$.

Because the statement itself is qualified by "**with probability at least** $1-\alpha$," the guarantee can never reach 100 % certainty unless $\alpha=0$ (which is impossible when sampling). As adversarial examples can exist outside the accepted confidence level, the statement "If the smoothed model has accuracy 100% and 'verified' 100% then no matter the attack it will have adversarial accuracy 100%" misinterprets this point.

### DeepPAC [2] also provides a **statistical** guarantee  

Definition 3.2 (page 3) of [2] introduces "$(\eta,\varepsilon)$-PAC-model robustness", i.e.  

>Let $f : \mathbb{R}^m \rightarrow \mathbb{R}^n$ be a DNN and $\Delta$ the corresponding score difference. Let $\eta, \varepsilon ∈ (0, 1]$ be the given error rate and significance level, respectively. The DNN $f$ is $(\eta, \varepsilon)$-PAC-model robust in $B(\hat{\boldsymbol{x}}, \boldsymbol{r} )$, if there exists a PAC model $\Delta \approx_{\eta,\varepsilon,\lambda} \Delta$ such that for all $\boldsymbol{x} \in B(\hat{\boldsymbol{x}}, \boldsymbol{r} ),\Delta(\boldsymbol{x}) + \lambda < 0$.


The parameters $\eta$ and $\varepsilon$ explicitly bound the **probability of failure**.  
Therefore DeepPAC does not offer a formal, sound-and-complete guarantee that no adversarial example exists; instead it offers the same kind of probabilistic guarantee as our method.


As for randomized smoothing, the required number of samples is strongly correlated with the standard deviation ($\sigma$
) and the certified radius. For instance, to certify a radius of 2/255 against noise with a standard deviation of $\sigma=1/(255*3)$, the randomized smoothing method would require an impractical number of samples (approximately $3.9×10^9$ ). This severely limits its application in real-world scenarios.


### The Probabilistic Guarantee and Distribution Assumptions about Our Method

Our method provides a probabilistic certificate of robustness. Specifically, it certifies with high confidence that the Intersection over Union (IoU) between the predicted bounding box and the ground truth will remain above a specified threshold with high probability, even under adversarial attack. For instance, if a bounding box is certified to have an IoU greater than 0.5, we can state with high statistical confidence that it is robust to object disappearance attacks relative to this detection criterion.

We **do not assume that the attacker follows any specific distribution**; rather, we provide a guarantee against the **worst-case perturbation within a given norm ball**. We only use uniform sampling to provide this guarantee, which is consistent with the Randomized Smoothing method (we have also added experiments with Gaussian sampling in our response to other reviewers).

The statement "a neural network can have accuracy 100%, adversarial accuracy to FGSM 0%, and yet this method will ‘verify’ 100% accuracy" is a misunderstanding. This scenario would not occur if an appropriate sampling distribution is used. As we have shown in our experiments for other reviewers, if a bounding box is certified as robust, it generally remains robust against stronger attacks like PGD.

If you are referring to providing a guarantee by sampling from a Dirac distribution as mentioned in your initial review, then **all probabilistic verification methods, including the Randomized Smoothing you mentioned, would reach the same conclusion of "verified 100% accuracy" for such a network**. This is a common issue for all probabilistic verification methods.


### Attack-based testing and sampling-based verification can never yield **complete** robustness guarantees  

To guarantee the complete elimination of adversarial examples, formal verification is required. However, this is a proven NP-complete problem, making it computationally infeasible for large-scale models like YOLO.

Any procedure that (i) relies on an attack heuristic or (ii) draws only finitely many samples can at best provide a *statistical* statement.
Our method is positioned in exactly this statistical regime, just like [1] and [2].

Given that providing absolute guarantees for large-scale object detectors is currently intractable, we have adopted a probabilistic verification approach. This method represents a pragmatic and meaningful compromise, extending formal verification principles to complex, real-world models for which exact certification is presently infeasible.

---

We hope that these clarifications remove the misunderstandings and put our contribution in the correct context relative to prior work. We remain grateful for your time and welcome any further discussion.