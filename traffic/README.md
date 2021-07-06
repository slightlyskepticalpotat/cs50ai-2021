I decided to initially base my model off the one shown in the lecture in the handwriting demo since I was familiar with that. After changing the `input_shape`, I trained the model and got `0.0557` accuracy after 10 epochs. Clearly, some changes were needed to increase the accuracy.

Step|
---|
All of the following were tested with the default testing parameters and the full gtsrb dataset (numbers may be slightly different in the demo).|
First, I tried adding a second convolutional layer identical to the first. This improved the accuracy to `0.9653`.|
Adding a third identical convolutional layer increased the accuracy to `0.9726`, but I decided not to add any more because every time I added a layer, the training time more than doubled.|
Adding a second pooling layer after the first convoluional layer decreased the accuracy to `0.9646`, but sped up the training.|
Moving the second pooling layer to be just after the first one increased the accuracy to `0.9795` and sped up the training. Win-win!|
I tested dropout rates from 0.5 to 0.8 (https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf), and 0.5 achieved the best accuracy of `0.9795`.|
Doubling the number of units in each convolutional layer decreases the accuracy to `0.0557`, I think this is because the model doesn't have enough data to work with.|
Adding another hidden layer (with the same number of units) decreased the accuracy to `0.9462` and increased the training time.|
Overall, I found that three sequential convolutional layers, two pooling layers, and one hidden layer with a dropout of 0.5 worked the best while keeping the training time reasonable, with an accuracy of `0.9795`.|

After, I compiled the model. I chose the `adam` optimiser because it's efficient, only requires a small amount of memory, and good for models with large amounts of data. (https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam) I also chose the `categorical_crossentropy` loss function because there are more than two label classes. (https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy). Finally, I decided to track our accuracy as the only metric.
