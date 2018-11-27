# **Behavioral Cloning** 

---

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network mimicking the nvidia model found here: ["End-to-End Deep Learning for Self-Driving Cars"](https://devblogs.nvidia.com/deep-learning-self-driving-cars) (model.py lines 40-51) 

The model uses Keras lambda layer (model.py line 41) to normalize the image data. Cropping (model.py line 42) of the top and bottom of each image is used to delete unnecessary data which may disrupt efficient training of the model.

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on images that were flipped on their horizontal (model.py line 22-27). This doubled the dataset and helped to generalize the model. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 53).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center-lane driving, driving the course in reverse, and recovering from the sides of the track. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to begin with a well-known covolutional network (Nvidia's) and to experiment with different types of training data.

Training data took many attempts to create a successful model. The model seemed to favor the right side of the track to the extreme of running into the right side of the bridge. To improve vehicle behavior, I found that course correction was more efficient training than center-lane driving.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. Honestly, I was a tad surprised to see the final result, cheering the vehicle along as it corrected off of the right boundary. The video of the successful run can be found at './run1.mp4'

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:

Layer  1: Lambda normalizes images and defines input shape.
Layer  2: Cropping layer removes unnecessary data from all images.
Layer  3: Conv2D 5 x 5 with 24 output layers, stride 2, and RELU activation
Layer  4: Conv2D 5 x 5 with 36 output layers, stride 2, and RELU activation
Layer  5: Conv2D 5 x 5 with 48 output layers, stride 2, and RELU activation
Layer  6: Conv2D 3 x 3 with 64 output layers, and RELU activation
Layer  7: Conv2D 3 x 3 with 64 output layers, and RELU activation
Layer  8: Flatten
Layer  9: Dense, fully connected, output 50
Layer 10: Dense, fully connected, output 10
Layer 11: Dense, fully connected, output 1

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded one lap on track one using center lane driving. Here are example images of center lane driving:

![alt text](https://github.com/blanklist/CarND-Behavioral-Cloning/blob/develop/center_1.jpg "center_1")
![alt text](https://github.com/blanklist/CarND-Behavioral-Cloning/blob/develop/center_2.jpg "center_2")

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct when it found itself near either side of the road. These images show what a recovery looks like starting from the right:

![alt text](https://github.com/blanklist/CarND-Behavioral-Cloning/blob/develop/from_right_1.jpg "from_right_1")

and from the left:

![alt text](https://github.com/blanklist/CarND-Behavioral-Cloning/blob/develop/from_left_1.jpg "from_left_1")

To augment the data sat, I also flipped images and angles thinking that this would help generalize the model. 

After the collection process, I had approximately 30,000 number of data points. More data points were not necessarily better. I attempted to train the model with more data, specifically with more center lane driving laps, which did not result in a successful run. It was not until I focused on training for the recovery from the left and right side of the track that the model succeeded a complete lap.
I then preprocessed this data with a normalization step (model.py line 41) and crop the top and bottom of the image (model.py line 42) to remove superfluous data. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 as evidenced by both testing and validation numbers decreasing for each epoch. I used an adam optimizer so that manually training the learning rate wasn't necessary.
