## Files ##
The project report is the TTT4275_Project file, which is a completion of the music classification problem.
Executable files are found int dist folder. One for each task. They can take up to a couple minutes to execute. 

The command execute the relevant code to generate the error rate and confusion matrix for Task 1

    cmd /K Part1.exe

The following command execute the relevant code to produce the histogram required in Task 2

    cmd /K Part2Histogram.exe

The following command execute the code to produce the error rates and confusion matrices for the testing of temeoving features in Task 2

    cmd /K Part2.exe

The following command execute the code to produce the confusion matrix and error rate for all 10 genres when replacing tempo with spectral_rolloff_var

    cmd /K Part3.exe

The following command execute the code to produce the error rate and confusion matrix for k-NN classifier with relative distance, and evaluating all available features.

    cmd /K Part4.exe
<!-- ## Template based classifiers ##
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
- Relative distance: $$d(x, ref_{ik}) = \frac{(x - \mu_{ik})}{\mu_{ik}}^T \frac{(x - \mu_{ik})}{\mu_{ik}}$$


# Training classifiers # -->


