## ECE 494 / CS 444: Deep Learning for Computer Vision, Fall 2026, Assignment 1

### Instructions
1.  Assignment is due at **11:59:59 PM on Thursday Sep 17, 2026**.

2.  See [policies](https://saurabhg.web.illinois.edu/teaching/cs444/fa2026/policies.html)
    on [class website](https://saurabhg.web.illinois.edu/teaching/cs444/fa2026).

3.  Submission instructions:
    1.  On gradescope assignment (Gradescope Code **44XPW2**) called `MP1-code`, upload your completed
    `models.py` and `featurize.py` files. These will be autograded and you will
    receive a score for your code.
        - Do not compress the files into `.zip` as this will not work.
        - Do not change the provided files names nor the names of the functions but
        rather change the code inside the provided functions and add new functions.
        Also, make sure that the inputs and outputs of the provided functions are
        not changed.
        - The autograder will give you feedback on how well your code did.
        - The autograder is configured with the python libraries noted in 
        `requirements.txt`. Autograding will fail if you use any packages that are not listed in requirements.txt and are not included by default with python.

    2. On gradescope assignment called `MP1-report`, fill out the text response
    to questions along with supporting figures and plots.

    3. We reserve the right to take off points for not following submission
    instructions. 

4.  Be careful not to work of a public fork of this repo. Make a
    private clone to work on your assignment. You are responsible for
    preventing other students from copying your work. Please also see point 2
    above.

5.  See [SUGGESTIONS.md](./SUGGESTIONS.md) for some suggestions for setup,
    workflow, and frequently asked questions.

### Problems

1. **Linear Algebra Review [4 pts Manually Graded].** Answer the following questions about
    matrices. Show the calculation steps (as applicable) to get full credit.

    1.1  **Matrix Multiplication [1 pts].** Let $`V = 
            \begin{bmatrix}
            -\frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\
                \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
            \end{bmatrix}`$ Compute
        $`V \begin{bmatrix} 1 \\ 0 \end{bmatrix}`$ and  
        $`V \begin{bmatrix} 0 \\ 1 \end{bmatrix}`$. What does matrix
        multiplication $`Vx`$ do to $`x`$?


    1.2  **Matrix Transpose [1 pts].**  Verify that $`V^{-1} = V^\top`$ What does
        $`V^\top x`$ do to $`x`$?


    1.3  **Diagonal Matrix [1 pts].**  Let $`\Sigma = 
            \begin{bmatrix}
                3 & 0 \\
                0 & 1
            \end{bmatrix}`$ Compute $`\Sigma V^\top x`$ where
        $`x = \begin{bmatrix} \frac1{\sqrt{2}} \\ 0 \end{bmatrix}, 
            \begin{bmatrix} 0 \\ \frac1{\sqrt{2}} \end{bmatrix}, 
            \begin{bmatrix} -\frac1{\sqrt{2}} \\ 0 \end{bmatrix}, 
            \begin{bmatrix} 0 \\ -\frac1{\sqrt{2}} \end{bmatrix}`$ respectively.
        These are 4 corners of a square. How is the square transformed by $`\Sigma V^\top`$ ?

    1.4  **Geometric Interpretation [1 pts].**  Compute $`A = U\Sigma V^T = \begin{bmatrix} -\frac{\sqrt{3}}{2} & \frac{1}{2} \\ -\frac{1}{2} & -\frac{\sqrt{3}}{2} \end{bmatrix} \begin{bmatrix} 3 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} -\frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \end{bmatrix}^\top`$. From the above
        questions, we can see a geometric interpretation of $`Ax`$: (1) $`V^\top`$
        first rotates point $`x`$ (2) $`\Sigma`$ rescales it along the
        coordinate axes, (3) then $`U`$ rotates it again. Now consider a
        general squared matrix $`B \in \mathbb{R}^{n\times n}`$ How would you obtain
        a similar geometric interpretation for $`Bx`$?

--- 

2. **Nearest Neighbors Classifier.**

    2.1 **[2 pts Autograded].** Complete the implementation of the
    `NearestNeighbor` class in [models.py](models.py). We will use L2 distance
    as the distance metric. You will need to complete
    the `train` and the `predict` functions. The input to the `predict` function
    is a *batch* of data that we want to make predictions for. 
    You are not allowed to use any external libraries or
    built-in functions that directly solve the problem for you (e.g. it is not
    ok to use `sklearn.neighbors.KNeighborsClassifier`, `cv2.knnMatch`, or
    `scipy.spatial.distance.cdist` among others). 
    
    *Hyperparameter selection:* The number of nearest neighbors `k` is a 
    hyper-parameter that you will need to select. Your implementation should 
    be general and handle different values fo `k` correctly. We found `k` values 
    of 1, 3 and 5 to be reasonably performant across the different settings. 

    *Testing your implementation:* You can test your implementation by running 
    the following command. This will use your implementation to make predictions 
    on the validation set and report the accuracy. 
    ```
    # Our implementation achieves an accuracy of about 64.07% on the validation
    # set in the following setting and takes about 2 seconds to run on a 4-core
    # 3.4GHz machine. 
    python demo.py --classifier knn --k 5 --num_train 100 --out_dir runs/knn/
    
    # Our implementation achieves an accuracy of about 85.79% on the validation
    # set in the following setting and takes about 3 seconds to run on a 4-core
    # 3.4GHz machine. 
    python demo.py --classifier knn --k 5 --num_train 1000 --out_dir runs/knn/
    ```
    Once you are happy with your implementation, you can run,
    ```
    python -m unittest tests.TestClassifier.test_knn_small 
    python -m unittest tests.TestClassifier.test_knn 
    ```
    This will evaluate your code on a number of other settings and confirm the
    results against the accuracies obtained by our implementation. We will be 
    running variants of these tests for autograding your code.

    *Hints:* Obvious ways to implement the `predict` function, say by using a
    for loop to go over the entire train / validation set, can be very slow. 
    You will benefit by vectorizing your code and avoiding any for loops to the
    extent possible. See [this tutorial](https://cs231n.github.io/python-numpy-tutorial/) to learn more about math
    and broadcasting operations on numpy objects. These are in-general
    significantly faster than writing for loops to iterate over the elements in
    a matrix. 
    
    At the same time naive vectorization that computes an intermediate
    $n_{train} \times n_{val} \times d$, where $d$ is the feature dimension (784
    in this case), will take too much memory. You will benefit from the
    following identity: $||a-b||^2_2 = (a-b)^T(a-b) = a^Ta + b^Tb -2a^tb$ and
    using the `np.dot` function. We would suggest to first focus on correctness
    (get the small tests, by running `python -m unittest
    tests.TestClassifier.test_knn_small` to pass first) and then worry about
    speed and memory. You can use your correct but inefficient code to confirm
    that your more efficient code works correctly.
    
    2.2 **[1 pts Manual Grading].** Next, we will visualize the nearest 
    neighbors for some sample digits. Complete the function 
    `get_nearest_neighbors` in `NearestNeighbor` to return the digit images
    corresponding to the  nearest neighbors from the training set for given
    test digit. You can use the plotting function `visualize_knn` in 
    [utils.py](utils.py) to visualize the nearest neighbors. Visualize the 10
    nearest neighbors for 5 random samples from the validation dataset in 2
    settings: a) when training dataset only has 100 samples and b) when training
    dataset has 10000 samples. Include the two generated visualization in
    submission to `MP1-report` and discuss what you observe.
    
---

3. **Multi-class Classification via Linear Regression.**
    In this question, we will develop a multi-class classifier for the MNIST
    digits using Linear Regression with a Regularizer term. 
    
    3.1 **[2 pts Manual Graded].** Let's first start with a 2 class linear
    regression model and see what we mean by regularization. Recall that for a
    training dataset with data points $x_i$ with labels $y_i$, (un-regularized)
    linear regression finds parameters $w$ by minimizing the following loss: 
    
    $$L_d = \frac{1}{N}\sum_{i=1}^{N} \left(w^Tx_i - y_i\right)^2$$
    
    It is common to add a L2 regularization term to this loss function:

    $$L_r = \lambda w^Tw$$
    
    where $\lambda$ is the regularization strength.

    The total loss is then given by:
    
    $$L = L_d + L_r$$.

    Following the derivation from class, derive a closed form solution for
    $w$ in terms of a data matrix $X$ (that stacks the data points
    $x_i^T$) and the label vector $Y$ (that stacks the labels $y_i$).

    3.2 **[2 pts Autograded].** Next, we will discuss the multi-class
    extension and you will implement it in python.

    One way of extending a 2-class classifier to a $K$-class classifier is to
    train $K$ 2-class classifiers ($f_k \forall k \in \[1 \ldots K\]$),
    where classifier $f_k$ classifies whether a given data point $x_i$
    belongs to class $k$ or not. To classify a test point $x$, we run all
    these classifiers on the point $x$ and use $argmax_k f_k(x)$ as the
    output class label. 

    Complete the definition of the `train` and `predict` functions in the
    LinearRegressionClassifier class in [models.py](models.py). Use the same
    regularization weight ($\lambda = $`reg_wt`) for all the 2-class classifiers. 
    When constructing labels $y_i$ for classifier for the $k$th digit, please
    set $y_i$ to $+1$ if the sample is of digit $k$ and to $-1$ if the sample
    is not of digit $k$. Please use the closed form solution you derived in 3.1.
    You may use numpy functions like `np.linalg.inv` to compute the inverse of
    a matrix, but don't use any external libraries or built-in functions that
    directly solve the problem for you (e.g. it is not ok to use
    `sklearn.linear_model.LinearRegression` or `pytorch`).
    
    Once you finish your implementation, you can train your classifier on the
    MNIST dataset using the following commands:
    ```
    # Takes about a second and achieves an accuracy of 61.16%. 
    python demo.py --classifier linear --wt 1e-3 --num_train 100 --out_dir runs/linear

    # Takes about a second and achieves an accuracy of about 83.95%.
    python demo.py --classifier linear --wt 1e-3 --num_train 10000 --out_dir runs/linear/
    ```
    
    Once you are happy with your implementation, you can run the following for a
    complete set of tests.
    ```
    python -m unittest tests.TestClassifier.test_linear
    ```

4. **Multi-class Classification via Logistic Regression.**

    4.1 **[2 pts Manually Graded].** In this problem, we will build multi-class 
    logistic regression classifiers to classify MNIST digits. Given a feature
    point $x$, the multi-class logistic regression classifier predicts the
    probability for each class $c$ as:
    
    $$p_c(x) = \frac{e^{w_c^Tx}}{\sum_{c'} e^{w_{c'}^Tx}} $$
    
    where $w_c$ is the parameter for class $c$. The classifier predicts the
    class with the highest probability. The parameters $w_c$ are learned by
    minimizing the cross-entropy loss as described below. Here, $x_i$ is the 
    data point from the training set and $y_i$ denotes the class it belongs to.
    
    $$L_d = -\frac{1}{N}\sum_{i=1}^N \log \left(p_{y_i} (x_i)\right)$$ 

    Typically, a regularization term, $L_r$, is also added to the loss function:
    
    $$L_r = \frac{\lambda}{2} \sum_{c=1}^C w_{c}^Tw_{c}$$
    
    where $\lambda$ is the regularization strength

    The total loss is then given by:
    
    $$L = L_d + L_r$$

    Show that the gradient of $L_d$ with respect to $w_j$ is given by
    
    $$\frac{\partial L_d}{w_j} = \frac{1}{N} \sum_{i=1}^N (p_j(x_i) - \delta_{y_i, j}) x_i$$
    
    where $\delta_{y_i, j}$ is the Kronecker delta. You should write down your
    derivation in the report PDF.

    4.2 **[4 pts Autograded].** Complete the multi-class logistic regression implementation of the `train`,
    `predict` and the associated helper functions in `LinearClassifier` class in
    [models.py](models.py). You will need to write code to compute the loss
    function and its gradient with respect to the parameters
    (`compute_loss_and_gradient`). You will also need to complete a training
    loop (`train`) that uses gradient descent to optimize the parameters. You
    are not allowed to use any external libraries or built-in functions that
    directly solve the problem for you (e.g. it is not ok to use
    `sklearn.linear_model.LogisticRegression` or `pytorch`).

    *Hyperparameter selection:* The regularization strength $\lambda$ is a 
    hyper-parameter that you can play with. We found a value around 1e-4 to work
    reasonably well across the different settings. $\lambda$ is typically varied
    in orders of magnitude (e.g. 1e-4, 1e-3, 1e-2, etc.).

    *Testing your implementation:* We have included a) some unit tests and 
    b) end-to-end performance metrics to help you test your implementation. 
    For the unit test, you can run the following command:
    ```
    # python -m unittest tests.TestClassifier.test_gradient_and_loss -v 
    ```

    For the end-to-end tests, you can run the following command:
    ```
    # Takes 5 seconds and achieves an accuracy of 71.28%.
    python demo.py --classifier logistic --lr 1e-1 --wt 1e-3 --num_train 100 \
        --out_dir runs/logistic/

    # Takes about 45 seconds to run on a 4-core 3.4GHz machine and achieves an accuracy of about 86%. 
    python demo.py --classifier logistic --lr 1e-1 --wt 1e-3 --num_train 1000 \
        --out_dir runs/logistic/
    ```
    We also provide reference loss values as a function of the number of
    iterations for the above two runs. You can access these using TensorBoard,
    using the following command: `tensorboard --logdir
    runs/logistic-reference/` and opening up the link that shows up. 

    Once you are happy with your implementation, you can run the following for a
    complete set of tests.
    ```
    python -m unittest tests.TestClassifier.test_logistic
    ```
    *Hint:* We will again suggest to first focus on correctness (get the unit
    test to pass first), and then work on optimizing your code. You can then
    compare the results of the two implementations to debug your optimized
    code.
    
    When computing the softmax, make sure you're using the [numerically stable
    softmax](https://jaykmody.com/blog/stable-softmax/)  so the intermediate
    values don't overflow.

    When computing the gradient, there are two important cases, When the class
    is the true class, then you'll want to update the weights for that class
    differently from when the class is not the true class. I would recommend
    working out what the derivative is by hand using the chain rule and then
    trying to implement it with simple for loops before vectorizing.

    4.3 **[1 pts Manually Graded].** Next, we will visualize the weights of the
    learned linear classifier. `demo.py` already saves the visualization of the
    learned weights to the specified out_dir folder as `w_vis.png`. Include the
    generated visualization in submission to `MP1-report` and discuss what you
    see. Do the weights correspond to the average shapes of the digits? Why or
    why not?
---

5. **Scaling Laws [2 pts]**. 

   In this problem, we are going to see how train and val losses and accuracies
   behave as we increase the amount of i) training data and ii) the size of the
   model. We will do this for the linear regression classifier, because that's
   much faster to work with. 

   You don't have to write any code for this problem, rather we will reuse code
   from Question 3 to generate some plots. For each setting below, you should:
   a) include the generated plot,
   b) describe what you see in the plot and any trends you observe, and 
   c) if you are able to, try to explain the trends you are seeing. Part (c) is
   not graded, but will prime you for when we actually discuss this in class in
   a few weeks.

   We will do this analysis in a slightly different setting that earlier
   questions. First, we will introduce a small amount of label noise in the
   training set, specifically 5% of the training samples have their label
   swapped to a different digit. Second, rather than classifying the raw
   pixels, we will first turn each image into a set of features using
   _rectified random projections_, and then train the linear classifier on
   those features. This way we can easily change the model size by simply
   changing the number of features we are using.

   A rectified random projection (RRP) from dimension $`d`$ to dimension $`W`$
   works as follows: Given a $`d`$-dimensional flattened image $`I \in \mathbb
   R^{d}`$ and projection vectors $`v_i \in \mathbb{R}^{d}`$ sampled from a
   normal distribution and $`a_i \in \mathbb{R}`$ (also sampled from a normal
   distribution), the $`i`$-th feature for this image $I$ is given by:

    $$x_i = \max(0, v_i^T I + a_i), \qquad i = 1, \dots, W.$$

    The $v_i$ and $a_i$ are sampled once, up front, and then not changed.
    Thus, the features are a fixed, precomputed function of the image, and the
    only thing we learn is the linear classifier sitting on top of them.
    In general, RRP features are a way to get a more effective representation
    of the data. Here, however, we use them as a way to easily change the model
    size $W$.

    There are two parameters of interest here. One is $`N`$, the number of
    training examples, and the other is $`W`$, the number of rectified random
    projections we use as features. We shall often call $`W`$ the parameter
    count, since the classifier has weights proportional to the number
    of features. We shall sweep each one independently, keeping the other
    fixed, and plot the training and validation error as a function of the
    varying quantity. 

    You can directly run [scaling_laws.sh](scaling_laws.sh) to directly
    generate the two plots: `sweep-features.png` and `sweep-samples.png`. The script calls
    [sweep.py](sweep.py) to generate the data and then calls
    [plot_sweep.py](plot_sweep.py) to generate the plots. You can look into
    these scripts if you are curious or want more details about the plots you
    are looking at. You can also modify the scripts to extract more information
    if you want or modify them if they don't work for you (but they should).
    Note that we are actually doing 5 runs and plotting average values so that
    we get to see more stable curves.

    5.1 **[1 pts Manually Graded] Effect of Increasing Training Set Size.**
    `sweep-samples.png` shows the train and val error (overlaid on the same
    plot) as a function of the number of training samples $`N`$ for each three
    different values of model parameters $`W \in \{200, 400, 800\}`$ shown in
    different colors. The training error is shown with dashed lines and the
    validation error is shown with solid lines. Note that the x-axis is
    logarithmic. 

    You should try to think about how the training and validation error change
    as we increase the number of training samples when keeping a fixed number
    of features. You can also think about how these trends change as we
    increase the number of features.

    5.2 **[1 pts Manually Graded] Effect of Increasing the Model Parameters.**
    `sweep-features.png` shows corresponding plots where the x-axis shows the
    number of model parameters and the different lines correspond to a
    different number of training samples. Once again look for trends in an
    individual train / val line and for trends across different train/val
    lines.

---

6. **ACCESS ID [1pt Manually Graded]**
    MP3, MP4, and MP5 will involve training neural network for different vision
    problems. You may benefit from having access to a GPU for these training.
    We have credits that can be used to get GPU compute hours on NCSA's Delta
    AI cluster. There is a 4 step sequential process your you to be able to use
    these GPU credits. You all will do Step 1 and Step 2, we will do Step 3,
    and then you will need to do Step 4. Each of these steps may take a few
    days of processing time, so we need to start now. As part of this MP, you
    will complete Step 1 and Step 2. After this MP is submitted, we will
    conduct Step 3. We will post an announcement on campuswire once we have
    completed Step 3 and provide instructions for Step 4 and also instructions
    for how to use the cluster.

    **Note:** Some of you may have GPU resources from elsewhere to use for your
    programming assignments. We still want you to follow through with these
    steps, just so that you have access to these GPU resources in case what you
    have turns out to not be sufficient. We want to batch the processing on our
    side and will not be able to accomodate one off requests later on. So,
    please follow through with these steps now, even if you think you won't
    need the GPUs later. 

    6.1. **Create an ACCESS ID with a complete profile. [0 pt]**
    Most of you probably don't have an ACCESS ID and will have to create
    one. Please follow instructions here
    https://operations.access-ci.org/identity/new-user to create an account.
    Whether you had an ACCESS ID or you just created one:
      - Make sure that your UIUC email address is associated with the ACCESS ID.
       We aren't able to add it to our ACCESS account if it isn't.
      - Make sure that your "Country of Residence" in set in your my ACCESS profile.
    
    6.2. **Provide us with your ACCESS ID [1 pt]** Once you create an ACCESS ID
    (or you had one from before) please go to
    https://allocations.access-ci.org/profile and look up your ACCESS ID. Even
    though you register using your UIUC identity, the ACCESS ID may be
    different from your UIUC net ID. Please provide us with this ACCESS ID in
    response to this question on GradeScope.
    
    6.3. **Course staff adds your ACCESS ID to our allocation [0pts].** You
    don't have to do anything for this step. We will take the information you
    submitted on gradescope and add you to our allocation.
    
    There is no way for us to give you access to GPU credits without your help
    in providing us with this information.
