# Q & As in Machine Learning

## Ensembles
#### 8. Thomas Dietrich describes 3 motivations for ensemble learning, what are these?
Statistical reason. A learning algorithm can be viewed as searching a space H of hypotheses to identify the best hypothesis in the space. By constructing an ensemble out of all of these accurate classifiers, the algorithm can average their votes and reduce the risk of choosing the wrong classifier.

Computational reason. When the training data is enough, the statistical problem is absent. But it may still difficult computationally for the learning algorithm to find the best hypothesis. An ensemble constructed by running the local search from different starting points may provide a better approximation to the true unknown function than individual classifiers.

Representational reason. In most applications of machine learning, the true function cannot be represented by any of the hypotheses in H. By forming weighted sums of hypotheses drawn from H, it may be possible to expand the space of representable functions. With a finite training sample, these algorithm will expand only a finite set of  hypotheses and they will stop searching when they find an hypothesis that fits the training data.

#### 9. What is the Bayes Optimal Classifier?
The Bayes Optimal Classifier is a classification technique. It is an ensemble of all the hypotheses in the hypothesis space. On average, no other ensemble can outperform it. It can be expressed with the following equation.

Where y is the predicted class, C is the set of all possible classes, H is the hypothesis space, P prefers to a probability, and T is the training data. As an ensemble, the Bayes Optimal Classifier represents a hypothesis that is not necessarily in H. The hypothesis represented by the Bayes Optimal Classifier is the optimal hypothesis in ensemble space.

#### 10. Your colleague has told you that she has implemented a Bayes Optimal Classifier. Should you believe her?
No. Because the Bayes Optimal Classifier cannot be practically implemented for any but most simple problems. There are several reasons why is cannot be implemented:
Most interesting hypothesis spaces are too large to iterate over, as required by the argmax.
Many hypothesis yield only a predicted class, rather than a probability for each class as required by the P(cj | hi).
Estimating the prior probability for each hypothesis (P(hi)) is rarely feasible.

#### 11. What is the difference between the bagging and boosting ensemble algorithms?
Their sample techniques for training each model in the ensemble are different.
Bagging uses a random sample of the original dataset. Each random sample is the same size as the dataset and sampling with replacement is used. In contrast, boosting pays more attention to instances that previous models misclassified. By doing this, It uses a weighted dataset. The weights for the misclassified instances are increased, and the weights for the correctly classified instanced are decreased.

When making predictions, bagging uses the majority vote or the median depending on the type of prediction required, whereas boosting uses a weighted aggregate of the predictions made by individual models. The weights used in this aggregation are the confidence factors.

#### 12. What is the key insight behind the gradient boosting algorithm?
In gradient boosting, the residuals ti - Mn-1(di) are treated as the negative gradients of the squared error loss function. So the key insight behind the gradient boosting is doing gradient descent on an error surface.

---

## Deep Learning Fundamentals
#### 13. Deep learning is often referred as “representation learning”. Why is this?
The goal of representation learning is to find an appropriate representation of data in order to perform a machine learning task.
In particular, deep learning exploits this concept by its nature. In a neural network, each hidden layer maps its input data to an inner representation that tends to capture a hider level of abstraction. These learnt features are increasingly more informative through layers towards the machine learning task like classification.

#### 14. What is the difference between a cost function and a loss function?
A loss function is a measure of the prediction error on a single training instance.
A cost function is the measure of the average prediction error across a set of training instances.
Cost functions allow us to add in regularisation.

#### 15. Describe how the dropout algorithm combats over-fitting in deep neural networks.


#### 16. Describe the gradient descent algorithm.
Choose random weights
Until convergence
    - Set all gradients to 0
    - For each training instance
    Calculate model output
    Calculate loss
    Update gradient sum for each weight and bias term
        - Update weights and biases using weight update rule


#### 17. Describe the back propagation of errors algorithm.
The key is the backpropagation of errors algorithm which allows us to propagate the gradient of the error from the output layer back through the earlier layers in a network.


#### 18. What is the difference between batch and stochastic gradient descent?
Stochastic gradient descent is easy to implement and can result in fast learning, but is computationally expensive and can result in a noisy gradient signal.
Batch gradient descent is computationally efficient and can result in a stable gradient signal, but requires gradient accumulation, can result in premature convergence. It can require loading large datasets into memory and can become slow.


#### 19. Some say that mini-batch gradient mixes the best of batch and stochastic gradient descent. Discuss.
Mini-batch gradient descent is relatively computationally efficient, does not require full datasets to be loaded into memory, and can result in a stable gradient signal, but requires gradient accumulation, and introduces a new hyper-parameter, mini-batch size.


#### 20. What are the exploding gradient and vanishing gradient problems?
When you training your deep networks, your derivatives get very very big or exponentially small.


#### 21. Training deep learning models is more likely to suffer from plateaus than local minima. Discuss.
DL chapter 8
Lower gradient really slows down progress
Algorithms like RMSprop, Adam etc help with this

---

## Deep learning CNNs & RNNs
#### 22. Why are convolutions so attractive for image processing tasks?


#### 23. What does it mean to say that a CNN has sparse connections?


#### 24. What does it mean to say that a CNN has shared weights?


#### 25. People often say that a CNN is translation invariant. What does this mean?


#### 26. Compare the gradient descent with momentum, RMSprop, and Adam optimisation algorithms.


#### 27. What does it mean to unroll an RNN?


#### 28. What is the differences between one-to-many, many-to-one, and many-to-many RNNs?


#### 29. Describe an application suited to each of a one-to-many, many-to-one, and many-to-many RNN?
