## Probability And Information Theory
---
### General stuff

*	**Difference between frequentists and bayesian:** frequentists are concerned with the rate of occurence of a certain event whereas Bayesians are concerned about the qualitative levels of certainty.

*	The **covariance** gives some sense of how much two values are linearly related to each other, as well as the scale of these variables:
	$$  Cov(\mathit{f(x)},\mathit{g(y)}) = \mathbb{E[(\mathit{f(x) - \mathbb{E[\mathit{f(x)}]}})(\mathit{g(y)}-\mathbb{E[\mathit{g(y)}]})]}  $$
	
	1. **High absolute values of covariance** mean that the **values change very much** and **both far from their respective means at the same time**
	2.	If the ** sign of the covariance is positive**, then **both variables** tend to take on relatively **high values simultaneously**
	3.	If the ** sign of the covariance is negative**, then **one variable** tends to take on a relatively **high value** at times that **the other** takes on a relatively **low value** and **vice versa**
	
	**N.B:** 
	
	a.	2 independent variables $$$ \to $$$ zero covariance
	b.	2 variables that have nonzero covariance $$$ \to $$$ dependent
	c.	Independence is a stronger requirement than zero covariance because it also exclude nonlinear relationships. But two variables can be dependent and have zero covariance.
	
	
---
	
### Common Probability Distributions

1.	**Bernoulli distribution** is a distribution over a single binary random variable.
2.	**Multinoulli Distribution** is a special case of the **multinomial distribution.** A multinomial distribution is the distribution over vectors in $$$\\{0,...,n\\}^k$$$ representing how many times each of the $$$k$$$ categories is visited when $$$n$$$ samples are drawn from a multinoulli distribution. Multinoulli distribution is a multinomial distribution where $$$n=1$$$
3.	**Gaussian Distribution** is the normal distribution. We choose it for 2 main reasons:

	*	The **central limit theorem** shows that the sum of many independent random variables is approximately normally distributed. i.e. many complicated systems can be decomposed into parts with more structure behaviour.
	*	Out of all possible probability distributions with the same variance, the normal distribution encodes the maximum amount of uncertainty over the real numbers, which means that it's a distribution that inserts the less amount of prior knowledge in the distribution.
4. **Exponential and Laplace Distributions**, in the context of deep learning, we often want to have a probability distribution with a sharp point at $$$ x = 0 $$$. LaPlace is a sharp peak at an arbitrary point $$$\mu$$$. That we can choose.
5. **The Dirac Distribution and Empirical Distribution**:
	we can think of the **dirac delta function** as being the limit point of a series of functions that put less and less mass on all points other than zero.
	
	The Dirac delta distribution is only necessary to define the empirical distribution over continuous variables. For discrete variables, the situation is simpler: an empirical distribution can be conceptualized as a multinoulli distribution
6.	**Mixtures of Distributions** are distributions defined by combining other simpler probability distributions.

---

### Technical Details of Continuous Variables

Suppose we have two random variables that are deterministic functions of one another. Suppose we have two random variables $$$\boldsymbol{x}$$$ and $$$\boldsymbol{y}$$$, such that $$$\boldsymbol{y} = g(\boldsymbol{x})$$$, where $$$g$$$ is an invertible, continuous, differentiable transformation. $$$p_y(\boldsymbol{y}) \ne p_x(g^{-1}(\boldsymbol{y}))$$$ This is a common mistake. The problem with this approach is that it fails to account for the distortion of space introduced by the function $$$g$$$.

---

### Information Theory

1.	We can quantify the amount of uncertainty in an entire probability distribution using the **Shannon entropy**.
2.	To measure how different two probability distributions are, we could use the ***Kullback-Leibler (KL) divergence***.