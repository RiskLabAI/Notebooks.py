## README.md

# Deep Quadratic Hedging with Neural Network Architectures

This repository contains the implementation and analysis of various neural network architectures for deep quadratic hedging of a European Call option with transaction costs under a discrete trading schedule. The work is based on the paper "An Analysis of Neural Network Architectures for Deep Quadratic Hedging" by Ahmad Aghapour, Hamid Arian, and Ali Fathi.

## Abstract

This project explores the deep stochastic optimal control (deepSOC) methodology to examine the model uncertainty resulting from the choice of neural network architecture for quadratic hedging of a European Call option with transaction costs. The paper demonstrates that parameterizing the hedge ratio policies at each time step by an independent neural network is more consistent with the dynamics of the gradients in the ADAM optimization and results in better training and higher performance.



### Quadratic Hedging

Quadratic Hedging (QH) in an incomplete market involves finding the initial capital \( w \) and the hedging strategy \( \theta^* \) that minimizes the mean-square hedging error (MSHE) at maturity. The mathematical formulation for QH is given by:

\[ E\left[L_T^2\right] = E\left[\left(C_T - w + \int_0^T \theta_u dS_u\right)^2\right]. \]

The optimal strategy \( \theta^*_t \) is given by:

\[ \theta^*_t = \frac{d\langle S, C \rangle_t}{d\langle S, S \rangle_t}. \]

Incorporating transaction costs, the option pricing equation becomes:

\[ C_t = w + \int_0^t \theta_u dS_u + L_t - \int_0^t \tau(S_u, \dot{\theta}_u, u) du. \]

### Deep Stochastic Optimal Control (deepSOC)

DeepSOC formulates the control problem as optimizing over a computational graph, where the control policies are parameterized by neural networks. The cost-to-go function to be minimized is:

\[ E[C] = E\left[\sum_{i=1}^{T-1} c_t(s_t, u_t(s_t | \theta_t)) + c_T(s_T)\right] =: L(\{\theta_t\}_{t=0}^{T-1}). \]

The pseudo-code for training the neural network computational graph is described in the paper.

## DeepSOC Architecture Analysis

### Equation for DeepSOC Objective Gradient

To understand the training dynamics of the deepSOC architecture, we analyze the gradients of the loss function with respect to the weight vector \( \theta_i \):

\[ E \left[ \left( \sum_{t=0}^{T-1} c_t(s_t, u_t(s_t | \theta_t)) + c_T(s_T) \right) \nabla_{\theta_i} \left( \sum_{t=i}^{T-1} c_t(s_t, u_t(s_t | \theta_t)) + c_T(s_T) \right) \Bigg| F_0 \right]. \]

### Observations

1. **Gradient Variance**: The gradients of different actions exhibit varying variances, with higher standard deviations at initial time steps, decreasing as the time to maturity approaches. This heteroskedasticity in the gradients impacts the training process, particularly when using the ADAM optimizer.

2. **Impact on ADAM Optimizer**: The ADAM optimizer adapts the learning rate to individual weights, performing larger updates for infrequent and smaller updates for frequent parameters. The heteroskedasticity in gradients means that some time steps may receive insufficient training, leading to suboptimal control policies.

3. **Independent Neural Networks**: By parameterizing each hedge ratio policy at each time step with an independent neural network, the training process becomes more consistent with the dynamics of the gradients, resulting in better performance.


## Implementation

The repository contains the following neural network architectures for implementing deepSOC:

1. **DeepSOC**: Each hedge ratio policy is parameterized by an independent feed-forward neural network (FFNN).
2. **DeepSOC with Weight Sharing**: Control policies for all time steps are parameterized by identical networks.
3. **Simple Feedforward**: A simple feedforward network architecture with specified layers and activation functions.

## Performance Analysis

The performance of each neural network architecture is evaluated based on the mean-square hedging error at maturity for different option maturities. The results indicate that deepSOC with independently parameterized control policies outperforms the other architectures, especially for longer maturities.

