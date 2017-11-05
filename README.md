#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report




## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* mynvidianet.py Implementation on the neural network by Nvidia as mentioned in the class
* myplainnet.py Implementation of a 1-layer neural network, that had been used fortesting the setup
* drive.py for driving the car in autonomous mode. Note that this as been modified from the original by increasing speed.
* model.h5 containing a trained convolution neural network
* hope1.mp4 An mp4 files shown a successful drive on track 1 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
(my code has been tested on a Windows PC)

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

NOTE: The implementation of the neural network itself is actually in mynividianet.py to keep the code modular.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of the Nvidia example neural network as shown in the classroom. During experimentation, some 
dropouts have been added to see if there was some improvement.

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 


####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Both the Udacity training data and custom recorded data was used for training and driving to check for relevant
differences. The approach was to have recordings for "Normal driving", some recovery driving and dedicated recordings for the bridge sections, since this caused problems in the beginning (car drove into bridge side.)

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to first setup a working end-to-end framework with an extremely simple neural network, to check all functionality (training, driving, simulator, video playbacks) and detect potential technical problems.

After the development steps were tested with the simple (fast) NN, the Nvidia network was used. The usual approach is to start with a network that seems simple enough to do the basic job and elaborate from that, so the Nvidia network was available and chosen as a candidate.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. Validation error was quite good from the beginning, but there were odd behaviours on certain sections of the car. 

I adressed these by using 
* data augmentation (flipped image, side cams)
* dedicated recording of special track sections

Validation results were fine, but there were still odd glitches with the car going of the track, when there were no
line markings. This indicated that there was a difference in training data and what the NN sees in actual driving. 

This was caused by the cv2 lib reading images in BGR format, and the drive.py providing images in RGB. The image processing
pipeline was changed for the training to convert the images to BGR with openCV and this resulted in drastic improvements (successful runs through the track)

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of the Nvidia NN from the class, with dropouts added to the fully connected layers.

####3. Creation of the Training Set & Training Process

I recorded data on track 1, using an analogue game controller, since using either keyboard or mouse did not result in smooth driving. 

I used data augmentation as mentioned in the class with flipping and additional cameras.

I recorded additonal data for recovery (i.e. the car going to the side of the track) and for the bridge  section.

I also recorded data from the second track.

Recorded data plus augmentation results in about 60K images + angles.


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3.I used an adam optimizer so that manually training the learning rate wasn't necessary.

####4. Track 2
I used the learned data on track two, however this result in the car going off track in the first dark section. Additonal data / augmentation would be necessary to train the car for darker tracks. I tried using image processing (histogram equalization) with unsufficient results.
