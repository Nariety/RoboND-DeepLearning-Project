[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

## Deep Learning Project ##

In this project, I trained a deep neural network to identify and track a target in simulation. So-called “follow me” applications like this are key to many fields of robotics and the very same techniques you apply here could be extended to scenarios like advanced cruise control in autonomous vehicles or human-robot collaboration in industry.

[image_0]: ./docs/misc/sim_screenshot.png

[model1]: ./docs/misc/model1.png
[model2]: ./docs/misc/model2.png
[model3]: ./docs/misc/model3.png

[2_Fol]: ./docs/misc/model2_FollowingHero.png
[2_Dis]: ./docs/misc/model2_HeroInDistance.png
[2_No]: ./docs/misc/model2_NoHero.png

[3_Fol]: ./docs/misc/model3_FollowingHero.png
[3_Dis]: ./docs/misc/model3_HeroInDistance.png
[3_No]: ./docs/misc/model3_NoHero.png

![alt text][image_0]

## Network Architecture and Performance##
In this project I used the architecture similar to the one shown in lesson: Encoder section, 1x1 Convolutional Layer, and Decoder section with skip connection between layers.

### **Model 1:** ![alt text][model1]
  **Final score was 38%.** Since this model didn't meet the pass grade, I didn't do much more analysis and moved on to a different model.

### **Model 2:**![alt text][model2]
  **Final score was 41%.** First passing grade achieved! The following images show the masks:
  * ![alt text][2_Fol]
  * ![alt text][2_Dis]
  * ![alt text][2_No]

### **Model 3:** ![alt text][model3]
  **Final score was 44%.** Second passing grade achieved! The following images show the masks:
  * ![alt text][3_Fol]
  * ![alt text][3_Dis]
  * ![alt text][3_No]

## Hyperparameters ##
**Optimal values that I found:**
* *learning rate = 0.01*
* *batch size = 32*
* *number of epochs = 30*
* *steps per epoch = 100*
* *validation steps = 50*
* *workers = 4*

#### learning_rate ####
I first started from a small value `0.0011`, which resulted in `val_loss = 0.4162` in the first Epoch. Then I changed the value to `0.01`, which resulted in `val_loss = 0.1188` in the first Epoch. And thus I chose `0.01` to be the learning rate.
#### batch_size ####
I first started from a bigger value `128`. However, as I changed the network architecture, it became too big for the GPU thus I decreased it. I first had a batch size of `64`, which resulted in `val_loss = 0.1145` in the first Epoch. Then I changed the value to `32`, which resulted in `val_loss = 0.1188` in the first Epoch while decreased the computation time significantly. Thus I chose `32` to be the learning rate.
#### num_epochs ####
As I was testing out different hyperparameters, I found that the `val_lost` reached a relatively stable score at about the `10` epoch. Even though increasing the number of epoch, the improvement was not very significant. However, to try to get a better score, I chose `30` to be the number of epoch.
#### steps per epoch ####
I chose `100` to be the number of steps per epochs because it provided a significant improvement in computation time without introducing much decrease in `val_lost`
#### workers ####
I chose `4` to be the number of workers because it provided a improvement without introducing much computation time.
#### validation steps ####
For the validation steps I kept the default values.

## Neural Network Layers and Techniques ##
In the context of this project, the fully convolutional network used consisted of three sections, encoder layers, 1x1 convolutional layer, and decoder layers.

The encoder layers are separable convolution layers that "squeeze" out features from images by building up the depth dimension. The first layer finds simple lines or colors and the outputs of the first layer is passed on as inputs of the second layer. As the encoder layers progress, more complex features are extracted.

The output of the encoder layers are then passed on to a 1x1 convolutional layer. It has a similar function to a fully connected layer, however, it not only flattens the image but also maintains spatial information.

The 1x1 convolutional layer is then connected to the decoder layers. The number and dimension of the decoder layers corresponds to those of the encoder layers. They upsample the extracted features to represent the spatial information in the original image dimension. They also use skip connection to concatenate layers from previous layers to fill pixels that were extracted out in the encoder section.

## Discussion ##
At the beginning of the project, I ran into a road block that one of the libraries used in `preprocess_ims.py` was deprecated and as a result I couldn't train the neural network with data that I recorded. I tried to fix it. Since my laptop does not have nearly enough processing power for neural network training, I had to rely on Amazon's `p2.xlarge` instance for training. I tried to update the outdated instance but because I could neither log in as `root` nor acquire the `sudo` permission right on the bat, I decided to move on to optimizing the hyperparameters and the architecture first instead. Fortunately, the second model worked reasonably well and was enough for a passing grade for this project.

Interestingly enough, even though from the looks of the masks generated in notebook, model2 seemed to have a better performance, model3 actually had a higher final score. Thus I'm submitting this project with settings for model3 (model1 and model2 are in notebook as well).

Even though the goal of this project is to train a neural network to recognize a "hero" human target, this fully convolutional network can be applied to recognize any other target such as dog, cat, car, etc. with corresponding training data sets.

## Future Improvement ##
I think more training data and possibly a more sophisticated architecture would improve the performance.

## Conclusion ##
Although in this project some of the guidances were not exactly in order and it was at times confusing for me to follow, it was a good introduction to deep learning. It was neither too complicated nor too superficial and it sparked my interest in deep learning and autonomous vehicles.
