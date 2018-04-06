# **Traffic Sign Recognition** 

This project is MaxEtzo's submission for **Traffic Sign Classifier** aassignment which is a part of [**Self-Driving Car Engineer Nanodegree's**](https://eu.udacity.com/course/self-driving-car-engineer-nanodegree--nd013) by **Udacity**

[//]: # (Image References)
[model]: ./resources/etzonet.png "Etzonet model diagram"
[hist_orig]: ./resources/histogram_original.png "Datasets distribution histograms"
[signs0-15]: ./resources/classes0-15.png "Random samples for sign classes 0 - 15"
[signs16-31]: ./resources/classes16-31.png "Random samples for sign classes 16 - 31"
[signs32-42]: ./resources/classes32-42.png "Random samples for sign classes 32 - 42"
[sign12_sym]: ./resources/class12_sym.png "Examples of symmetry transformations on 'Priority road' sign"

**Build a Traffic Sign Recognition Project**
The goals of this project are the following:
* Load the [data set](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip) (German Traffic Sign Benchmark Dataset)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report 


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

### Dataset Summary & Exploration

#### 1. Basic summary of the dataset:

I used python's native `len` and numpy's `shape` methods to obtain the information on the datasets and the format.
```python
n_train = len(y_train)
n_valid = len(y_valid)  
n_test = len(y_test)  
image_shape = X_train.shape[1:4]  
n_classes = len(set(y_train))
```

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Below are distribution histograms for training, validation and test sets respectively. 

![hist_orig]

All, training, validation, and test sets have relatively same signs distribution. It must be noted that sets are biased toward certain signs. 

* Most frequent sign in training set - *'Speed limit (50km/h)'* with 2010 samples
* Least frequent sign in training set - *'Speed limit (20km/h)'* with 180 samples

Ratio of the most frequent sign to the least one is over 10! On the one hand, it's not an ideal situation as a model based on such dataset will be **biased (i.e. perform better) towards over-represented signs**. On the other hand, based on the mere size of the datasets, one may conclude that **these distributions somewhat represent real distributions of the signs on the roads**! In fact, in five years I have never encountered *'Speed limit (20km/h)'* sign in Germany. Whereas 30 and 50 km/h signs are common due to 'residential' and 'urban' speed limits regulations. Some people argue that datasets must be equalized in order to avoid biasing, and others going for integrity of the distribution solely for achieving higher accuracy. IMO, the golden line is somewhere in between. 

Let's have a look at the random samples for each sign class:
![signs0-15]
![signs16-31]
![signs32-42]
as can be observed, datasets comprise of images of wide range of quality. Some signs are hardly readable as they are either too dark, too bright and/or too blur. Also, in some samples signs are partially visually obstructed, covered with shadows, or contain parts of other signs.  

### Design and Test a Model Architecture
#### 1. Dataset augmentation.
We can further increase dataset by applying simple geometrical transformations. 
##### a. Symmetries: 
Certain signs are horizontally, vertically, diagonally (*-pi/4* and *pi/4*) and/or rotationally (*pi/2*, *pi*, *3pi/2*) symmetrical. For example, all aforementioned transformations can be applied to *'Priority road'* sign, while *'Bumpy road'* only horizontally symmetrical.
![class12_sym]
Certain sign pairs (typ. *'left-right'* type of signs) are horizontal mirrored copies. So by applying the transformation sample set of respective counterpart sign can be extended.

After applying symmetry extension of the training dataset, histograms look as follows
##### b. Perspective transformation:    
#### 2. Data preprocessing 
http://forums.fast.ai/t/rgb-vs-hsv-colorspace/1885/2
https://www.reddit.com/r/MLQuestions/comments/6ueiis/hsv_vs_rgb_grayscale_for_convolutional_neural/
https://www.reddit.com/r/MLQuestions/comments/6rfpzj/does_the_color_basis_matter_in_a_cnn/
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3254613/
As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


#### 3. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 4. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.
temp: https://www.youtube.com/watch?v=_P-NZmjl5gM

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?