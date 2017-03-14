**Behavioral Cloning**

# Writeup Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./cnn-architecture.png "NVIDIA architecture"
[image2]: ./figure1.png "Model Visualization"
[image3]: ./example_center.jpg "Example image from the center camera"
[image4]: ./example_left.jpg "Example image from the left camera"
[image5]: ./example_right.jpg "Example image from the right camera"
[image6]: ./example_left_flipped.jpg "Example flipped image from the left camera"
[image7]: ./example_center_flipped.jpg "Example flipped image from the center camera"
[image8]: ./example_right_flipped.jpg "Example flipped image from the right camera"
[image9]: ./training_results.png "Results of training the model"

# Rubric Points
## Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
## Files Submitted & Code Quality

### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results
* run1.mp4 being a video of the successful autonomous drive

### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## Model Architecture and Training Strategy

### 1. An appropriate model architecture has been employed

My model is on constructed on lines 59-73.

The architecture is essentially the NVIDIA architecture (a Convolutional Neural Network architecture) recommended in the Udacity lessons (from here: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). There are 3x3 kernels and 5x5 kernels in the convolution layers and depths ranging from 64 to 24. I added drop outs in between the first three dense layers to make the model more robust (i.e. combat overfitting).

Here is the NVIDIA model:

![alt text][image1]

### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 69 and 71).

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 108). The number of epochs I went with was 3, as there seemed to be no added value in more. Additionally, I went with a batch size of 32 given fairly good results from that.

### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the Udacity dataset, which I learned has mostly straight driving and less recovery situations.

For details about how I created the training data, see the next section.

## Model Architecture and Training Strategy

### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to take the NVIDIA model and improve it to the point where driving around the track in simulation mode would be acceptable. The only modifications were a lambda layer for normalization (if you consider normalization a modification) and dropout layers. As for dropout layers there are many permutations (i.e. how many of them, where to put them, what rate). I don't doubt that there are more optimal configurations with respect to dropout.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that after two epochs the validation error was not improving.

To combat the overfitting, I modified the model by adding drop out.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. Originally I used my own trained data set. After grasping with the fact that I am no longer that great at video games, especially with ones using a mouse or arrows to drive a car, I decided to use the Udacity data set. On my first attempt the simulation on the trained set was much better. To improve this I played around with dropout layers, batch size, epochs and correction.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

### 2. Final Model Architecture

The final model architecture (model.py lines 59-73) consisted of a convolution neural network with the following layers and layer sizes:

![alt text][image2]

### 3. Creation of the Training Set & Training Process

As mentioned, after being discontent with my own training data, I switched to the Udacity training set given on the website. Here is an example image of center lane driving:

![alt text][image3]

And an example of an image from the left and right cameras:

![alt text][image4]
![alt text][image5]

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here are three images that has then been flipped, from the left, center and right cameras respectively.

![alt text][image6]
![alt text][image7]
![alt text][image8]

After the collection process, I had 38568 data points. I then preprocessed this data by normalizing the pixels centered around 0 +/- 1. This was conveniently done with a lambda layer, which took care of the normalization both the training and running the model in just one line.


I finally randomly shuffled the data set and put 10% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by trying different epochs and comparing training loss vs validation loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][image9]
