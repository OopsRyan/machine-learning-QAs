# Q & As in Machine Learning

## General
#### 1. What is the difference between supervised learning and unsupervised learning?
The primary difference is the data used in either method of machine learning.
The input data used in supervised learning is labelled. This means that the machine is to determine the hidden patterns from already labelled data.
However, the data used in unsupervised learning is not labelled. The machine has to categorize and label the raw data before determining the hidden patterns of the input data.

#### 2. Inductive machine learning is often referred to as an ill-posed problem. What does this mean?
It is possible to find many models that are consistent with a given training set. There is typically not enough information in the training data to choose a single best model, so inductive machine learning is referred to as an ill-posed problem.

#### 3. How will GDPR impact on the use of machine learning?
There are three subjects that might impact the use of machine learning.
General data protection. GDPR concerns personal data only. Machine learning can not be used to identity a person.
Prohibition on profiling and automated decision making. Prohibiting only when the decisions produce legal effects or similar significant effects.
Right to explanation. If legally binding, the explanation of specific decisions is required.


#### 4. What is the difference in evaluation approaches for machine learning for industry and machine learning for academic research?
From the industry point view, the evaluation emphasizes the evaluating a model that we would like to deploy for a specific task.
From the academic research point view, the evaluation emphasizes the comparing machine learning methods. Lots of research evaluations reduce to a benchmark across multiple methods on multiple datasets.
One of the key differences is the need for significance testing.

#### 5. If performing a machine learning benchmark of 10 algorithms using 10 datasets what approach would you take and what statistical significance tests would you use?
Friedman aligned rank test to first test whether a significant difference between the performance of the algorithms over the datasets exists.
If a difference does exist, then a pairwise Nemenyi test should be performed to show between which algorithm pairs the significant differences exist.

#### 6. Why are there so many different evaluation metrics (e.g., AUC, accuracy, F1 score, gain, lift, ...) used in machine learning?
Different evaluation metrics are good at measuring performance for different problems. For example, macro averaging metric is better than micro averaging when classification datasets are imbalanced.

#### 7. Machine learning is plagued by hyper-parameters set with magic numbers. Discuss.
Hyper-parameters are values we set on the model before training. Searching for the best hyper- parameter can unlock the maximum potential of the model.
Machine learning is plagued by hyper-parameters because sometimes it is difficult to select the best hyper-parameter for the problem.
The default hyper-parameters may be not inappropriate for a specific problem. We can apply Grid search, random search, and Bayesian optimization to do hyper-parameter optimization.

---

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
During training randomly drop out units according to a dropout probability at each training epoch. Scaling up output of remaining units by dividing by dropout probability, which ensures that overall activations remain in the same range throughout training process.
When making predictions, no dropout applied.

#### 16. Describe the batch gradient descent algorithm.

    Choose random weights
    Until convergence
        Set all gradients to 0
        For each training instance
        	Calculate model output
        	Calculate loss
        	Update gradient sum for each weight and bias term
        Update weights and biases using weight update rule


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
Because the number of dimensions are so large in deep learning models, the probability that an optimum only consists of a combination of minima is very low. This means 'getting stuck' in a local minimum is rare. Intuitively, in a single dimension, it is easy to obtain a local minimum by tossing a coin and getting heads once. In n-dimensional space, it is exponentially unlikely that all n coin tosses will be heads.

---

## Deep learning CNNs & RNNs
#### 22. Why are convolutions so attractive for image processing tasks?


#### 23. What does it mean to say that a CNN has sparse connections?
In traditional neural networks, every output unit interacts with every input unit. However, CNNs typically have sparse interactions. This is accomplished by making the kernel smaller than the input. For example, when processing an image, the input image might have thousands or millions of pixels, but we can detect small, meaningful features such as edges with kernels that occupy only tens or thousands of pixels. This means that we need to store fewer parameters, which both reduces the memory requirements of the model and improves its statistical efficiency. So CNNs have sparse connections.


#### 24. What does it mean to say that a CNN has shared weights?
In traditional neural networks, each element of the weight matrix is used exactly once when computing the output of a layer. It is multiplied by one element of the input and then never revisited. In CNNs, each member of the kernel is used at every position of the input (except perhaps some of the boundary pixels). The weight sharing used by CNNs means that rather than learning a separate set of weights for every location, we learn only one set for all locations. It can further reduce the storage requirements of weights. This can also reduce the memory requirements, and improve statistical efficiency.


#### 25. People often say that a CNN is translation invariant. What does this mean?
Invariance to translation means that if we translate the input by a small amount, the value of outputs does not change. Invariance to local translation can be a very useful property if we care more about whether some feature is present than exactly where it is.

#### 26. Compare the gradient descent with momentum, RMSprop, and Adam optimisation algorithms.
Momentum applies an exponentially weighted moving average to gradient descent. Weight update rule uses averaged gradients.
RMSprop modifies gradients by scaling by exponential average of square of gradients. Weight update rule uses modified gradients.
Adam adds bias-correlation and momentum to RMSprop. Its bias-correction helps Adam outperform Momentum and RMSprop towards the end of optimization as gradients become sparser.

#### 27. What does it mean to unroll an RNN?
Unrolling an RNN means that we write out the network for the complete sequence. For example, if the sequence we care about is a sentence of 5 words, the network would be unrolled into a 5-layer neural network, one layer for each word.

#### 28. What is the differences between one-to-many, many-to-one, and many-to-many RNNs?
The differences between one-to-many, many-to-one, and many-to-many RNNs are the number of input units and outputs units. For example, a one-to-many RNN has one input unit and many output units.

#### 29. Describe an application suited to each of a one-to-many, many-to-one, and many-to-many RNN?
One to many: image captioning takes an image and outputs a sentence of words.
Many to one: sentiment analysis where a given sentence is classified as expressing positive or negative sentiment.
Many to many: machine translation: an RNN reads a sentence in English and then output a sentence in Chinese.