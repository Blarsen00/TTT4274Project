# Classifiers #

## MAP Classifiers ##
### Bayes Decision Rule ###
BDR also called a MAP classifiers has the following decision rule:

$$x \in w_j \Leftrightarrow P(w_j/x) = \max P(w_i/x) \forall i \in [1, C]$$

Which essentially means that the input $x$ is classified to the class $w_j$ if $x$ has the highest probability to be in $w_j$ assuming the probability $p(w_j/x)$. 

### Plug-in-MAP classifier ### 
Plug-in-MAP is a MAP classifier which assume that the probability distribution is the Gaussian distribution:

$$p(x/w_i) = N(\mu_i,\Sigma_i) = \frac{1}{\sqrt{2\pi}^D |\Sigma_{ik}|}\exp(-\frac{1}{2}(x-\mu_i)^T \Sigma_{i}^{-1} (x-\mu_i))$$

### Gaussian Mixture Model (GMM) ###
We also have the Gaussian Mixture Model (GMM) which is a mixture of Gaussian distributions
    to better approximate the actual distribution, as a pure Gaussian model would often be inaccurate.
    This model is a sum of Gaussians as such:
    $$p(x/w_i) = \sum_{k=1}^{M_i}c_{ik} N(\mu_{ik},\Sigma_{ik}) = \sum_{k=1}^{M_i}
    \frac{c_{ik}}{\sqrt{2pi}^D |\Sigma_{ik}|} \exp(-\frac{1}{2}(x-\mu_{ik})^T \Sigma_{ik}^{-1} (x-\mu_{ik}))$$


$P(w_i)$ must be estimated as well as the densities, but it is not as critical as the. 
of the problem 

## Linear Classifiers ##
Linear classifiers are discriminant classifiers where each class is described by the following discriminant function: 
$$
\begin{align*}
    x &\in w_j &&\Leftrightarrow& g_j &= \max g_i(x) &&\text{where} & g_i(x) &= w_i^T x + w_{io}
\end{align*}
$$

## Template based classifiers ##
Template based classifiers match the input x towards a set of references (templates) which have the same form as x. The decision rule finds the reference which is closest to the input and assigns it the same class. This method is called the "nearest neighbor - (NN)". Distinguishing factors between template based classifiers are 

- Decision rule (such as NN)
- Distance measure between input and references
- Choice of reference

### Decision rule KNN ###
Find the closest K references to input x, and the class with most references classify the input. If there are two classes with the most references, x get classified in the class with the closest reference.

### Distance measures ###
$ref_{ik} = (\mu_{ik}, \Sigma_{ik})$
- The Mahalanobis distance: $$d(x, ref_{ik}) = (x - \mu_{ik})^T \Sigma_{ik}^{-1} (x - \mu_{ik})$$
- Euclidian distance: $$d(x, ref_{ik}) = (x - \mu_{ik})^T (x - \mu_{ik})$$


# Training classifiers #

## Plug-in-Map using Most Likelihood (ML) based training ##
The ML algorithm estimates the parameters for a single class at a time. We have a training subset $X_{N_i} = \{x_{i1}, ...,x_{iN_i}\}$ where all inputs are independent and belong to the same class $w_j$, and the input parameters $\Lambda_i = \{\mu_i, \Sigma_i\}$ that we want to estimate. The likelihood for the subset is given by:

$$L[X_{N_i}, \Lambda_i] = \prod_{k=1}^{N_i}p(x_{ik}/\Lambda_j)$$

When we assuma a exponential distribution such as the Gaussian distribution we can simplify the equation by using the loglikelihood in its place:

$$LL[X_{N_i}, \Lambda_i] = \sum_{k=1}^{N_i}\log[p(x_{ik}/\Lambda_j)]$$


In order to maximize the function we find the gradient and then take the parameters that make the gradient equal zero.

$$\nabla_{\Lambda_i} LL[X_{N_i}, \Lambda_i] = \sum_{k=1}^{N_i}\nabla_{\Lambda_i} \log[p(x_{ik}/\Lambda_j)] = 0$$

### ML training single Gauss case ###

For the single Gauss case we find that the logarithm of the distribution is as follows:

$$ \log[p(x_{ik}/\Lambda_i)] = -\frac{D}{2}\log(2\pi)-\log(|\Sigma_i|) - \frac{1}{2}(x-\mu_i)^T \Sigma_i^{-1}(x-\mu_i)$$

When taking the gradient with respect to $\mu_i$ and $\Sigma_i$, we can derive with the following estimators for the mean and covariance:

$$
\begin{align*}
    \hat \mu &= \frac{1}{N_i} \sum_{k=1}^{N_i} x_{ik} 
    % &&\text{and}&
    &&& 
    \hat \Sigma &= \frac{1}{N_i} \sum_{k=1}^{N_i} (x_{ik}-\hat\mu_{i})^T(x_{ik}-\hat\mu_{i})
    \end{align*}
$$

### Ml training using GMM densities ###

Taking the gradient of the GMM gives us an equation with the estimate on both sides of the equal sign, $\Lambda_i = f(\Lambda_i)$. In order to solve this we use the algorith Expectation Maximation - EM which will find a (suboptimal) solution. With a set of initial estimates $\Lambda_i(0)$ we have the following algorithm:

$$ 
\begin{align*}
    \Lambda_i(m) &= f(\Lambda_i(m-1)) &&\Leftrightarrow& LL(m) &> LL(m-1) & m&=1,2,...
\end{align*}
$$ 

The algorithm stops when the improvement in $LL$ gets better than a chosen value.

## Means square error based training on linear classifiers ##

The output of the linear classifier is given by the putput vector $g=Wx+w_o$ with dimenson $C$. This expression can be rewritten as $g = [W \ w_o][x^T \ 1]$ and we can redefine the input and matrix to be $[W \ w_o] \rightarrow W$ and $[x^T \ 1] \rightarrow x$ so we get $g=Wx$. $\\$
The linear classifier is not statistically based, so we use the mean square error (MSE) as an optimization criteria. 

$$
$$