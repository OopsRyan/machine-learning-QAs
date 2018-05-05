# Q & As in Machine Learning

## Ensembles
### 8. Thomas Dietrich describes 3 motivations for ensemble learning, what are these?
Statistical reason. A learning algorithm can be viewed as searching a space H of hypotheses to identify the best hypothesis in the space. By constructing an ensemble out of all of these accurate classifiers, the algorithm can average their votes and reduce the risk of choosing the wrong classifier.

Computational reason. When the training data is enough, the statistical problem is absent. But it may still difficult computationally for the learning algorithm to find the best hypothesis. An ensemble constructed by running the local search from different starting points may provide a better approximation to the true unknown function than individual classifiers.

Representational reason. In most applications of machine learning, the true function cannot be represented by any of the hypotheses in H. By forming weighted sums of hypotheses drawn from H, it may be possible to expand the space of representable functions. With a finite training sample, these algorithm will expand only a finite set of  hypotheses and they will stop searching when they find an hypothesis that fits the training data.

### 9. What is the Bayes Optimal Classifier?
The Bayes Optimal Classifier is a classification technique. It is an ensemble of all the hypotheses in the hypothesis space. On average, no other ensemble can outperform it. It can be expressed with the following equation.

Where y is the predicted class, C is the set of all possible classes, H is the hypothesis space, P prefers to a probability, and T is the training data. As an ensemble, the Bayes Optimal Classifier represents a hypothesis that is not necessarily in H. The hypothesis represented by the Bayes Optimal Classifier is the optimal hypothesis in ensemble space.

### 10. Your colleague has told you that she has implemented a Bayes Optimal Classifier. Should you believe her?
No. Because the Bayes Optimal Classifier cannot be practically implemented for any but most simple problems. There are several reasons why is cannot be implemented:
Most interesting hypothesis spaces are too large to iterate over, as required by the argmax.
Many hypothesis yield only a predicted class, rather than a probability for each class as required by the P(cj | hi).
Estimating the prior probability for each hypothesis (P(hi)) is rarely feasible.

### 11. What is the difference between the bagging and boosting ensemble algorithms?
Their sample techniques for training each model in the ensemble are different.
Bagging uses a random sample of the original dataset. Each random sample is the same size as the dataset and sampling with replacement is used. In contrast, boosting pays more attention to instances that previous models misclassified. By doing this, It uses a weighted dataset. The weights for the misclassified instances are increased, and the weights for the correctly classified instanced are decreased.

When making predictions, bagging uses the majority vote or the median depending on the type of prediction required, whereas boosting uses a weighted aggregate of the predictions made by individual models. The weights used in this aggregation are the confidence factors.

### 12. What is the key insight behind the gradient boosting algorithm?
In gradient boosting, the residuals ti - Mn-1(di) are treated as the negative gradients of the squared error loss function. So the key insight behind the gradient boosting is doing gradient descent on an error surface.
