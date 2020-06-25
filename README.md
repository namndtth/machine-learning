A big project for studying machine learning. Here, I want to implement things from scratch using numpy library to gain deeper understanding about ML/AI.

- The 1st project must be linear regression. In this project:
  - I assume our hypothesis is h_theta(x) = w0 * x0 + w1 * x1 (w0 is bias).
  - I implement:
    - Forward propagation, which is feed input X into our model.
    - Calculate cost function over m training examples: 
      - $L(h_{\theta}(x), y)=\frac{1}{2 * m} * (h_{\theta}(x)-y)^2$

    - Backward propagation: in this step, i calculate the derivative of cost function with respect to our parameters W.
      - $\frac{dL}{dW_0}=\frac{1}{m} * x_{0}*(h_{\theta}(x)-y)$
      - $\frac{dL}{dW_1}=\frac{1}{m} * x_{1}*(h_{\theta}(x)-y)$
    - Vectorize implementation:
      -  $\frac{dL}{dW}=\frac{1}{m} * X * (h_{\theta}(X)-y)$
    - Using gradient descenst algorithm to update our parameteres.
      - $W = W - \alpha * \frac{dL}{dW}$
    - Using numerical gradient for gradient checking:  
        $\frac{J(\theta+) - J(\theta-)}{(2 * \epsilon)}, \epsilon=1e-4$