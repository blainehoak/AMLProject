import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision as tv


""" 
transform is a chain of image transformations, joined together by the Compose method
Conpose takes a list of transforms and composes them, the transforms included here are:
    ToTensor - which will take in an image and calculate the RGB value for each pixel,
                then normalize the values between 0 and 1, returning a tensor
    Normalize - takes in a tuple of size n for n channels where the first tuple contains the 
                mean for each channel and the second tuple contains the standard dev
                normally done with data whose features have different ranges, but this is
                helpful here because if there were any outliers that caused a compression of the 
                distribution, the Normalize method will help revive that shape by centering the data at
                0.5 (halfway between 0 and 1) and having all the data fall within 1 standard deviation of
                the mean 0.5 +- 0.5 = 0, 1. 
"""
transform = tv.transforms.Compose(
    [
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5,), (0.5,)),
    ]
)

# download all the datasets and transform them
trainset = tv.datasets.MNIST(
    "~/Documents/Grad/AMLProject", download=False, train=True, transform=transform
)
valset = tv.datasets.MNIST(
    "~/Documents/Grad/AMLProject", download=False, train=False, transform=transform
)

# loading the data, batch size is the number of images we want to read in one go
# TODO: why do we set shuffle equal to True?
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

#### Begin exploratory data analysis section ####

"""
Getting an iterator over the dataset, which, when we call the next() method, will return a tuple containing:
    images - a tensor with 4 dimensions:
            number of images per batch
            number of batches
            image width in pixels
            image height in pixels
    labels - the true labels for each image in this batch
"""

"""UNCOMMENT FOR DATA ANALYSIS start

dataiter = iter(trainloader)
images, labels = dataiter.next()

figure = plt.figure()
num_of_images = 60
for index in range(1, num_of_images + 1):
    plt.subplot(6, 10, index)
    plt.axis("off")
    plt.imshow(images[index].numpy().squeeze(), cmap="gray_r")
plt.show()

UNCOMMENT FOR DATA ANALYSIS end"""


#### Building the Neural Network ####

input_size = 784  # just 28x28
hidden_sizes = [128, 64]  # TODO:why 2 hidden layers and why these numbers?
output_size = 10  # confidence scores for classes (digits) 0-9

"""
Creating the model - the output of each layer must be equal to the input of the next layer. 
We define the activation function for each layer after defining the layer.
"""
model = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_sizes[0]),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_sizes[0], hidden_sizes[1]),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_sizes[1], output_size),
)

#### Training Process ####

"""
Before the backward pass, the model weights are initialized to None. Once we call the backward() method, the weights get updated and as this happens
more and more times, the training loss gradually decreases. So, we want to repeat this process over all the epochs of the training data.
We will use stocastic gradient descent as our optimizer (TODO:why?) but there are other options (e.g. Adam)

We also need to decide what the hyperparameters(learning rate and momentum) should be. TODO: test with different learning rates? possibly different optimizers?

Now we have to define how we will calculate the loss of our system so that we can preform backpropogation
Here we will use cross entropy loss (log likelihood loss) to do this. TODO: test with other cost functions
"""
criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
epochs = 15
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:

        """
        We can get the next iteration of the dataset (next batch) and since images is a tensor containing all the pixels of every image
        in each batch, we can resize it so that each row (the 1st dimension) contains an image and each column (2nd dimension) contains the data for that image.
        By passing in -1 as the second argument, we are telling it to infer what the second dimension should be in order to keep all the data
        and have (in this case) 64 rows. So it should infer the second dimension to be 784.
        """

        images = images.view(images.shape[0], -1)

        optimizer.zero_grad()

        # now that we have reshaped the input, we can pass the first batch through the model and calculate cross entropy loss
        output = model(images)
        loss = criterion(output, labels)

        # Model learning through backprop
        loss.backward()

        # Optimizes/updates weights
        optimizer.step()

        # Keeping track of the loss as more images are processed
        running_loss += loss.item()

    print("Epoch {} - Training loss: {}".format(e, running_loss))
    # print(output)
    # print(labels)

#### Begin Evaluation ####
"""
Here, we will evaluate the accuracy of our trained model on the evaluation set. TODO: k-fold cross val and use the test on the val set.
We can iterate over everything in the valloader and go one image at a time to see if the prediction was correct.
For each image, we can resize it to be a vector containing all the pixels, just like we did for the training data.
"""
correct_count, all_count = 0, 0
for images, labels in valloader:
    for i in range(len(labels)):
        img = images[i].view(1, 784)

        with torch.no_grad():
            output = model(img)

        # getting the predicted value, which is the max value from the output
        pred_label = torch.argmax(output)
        true_label = labels[i]

        # do comparison and tally if the prediction was correct
        if true_label == pred_label:
            correct_count += 1
        all_count += 1

print("Number of Images Tested:", all_count)
print("Model Accuracy:", (correct_count / all_count))
og_accuracy = correct_count / all_count

#### Adversarial Example Generation ####

""" Starting off with just an FGSM attack. For this attack we need:
        - the image to perturb
        - a value for epsilon
        - the gradient of the loss function wrt the input image
"""

saved_ex = []
correct_count, all_count = 0, 0
for images, labels in valloader:
    for i in range(len(labels)):
        img = images[i].view(1, 784)

        # need this for generating adversarial examples
        img.requires_grad = True

        # with torch.no_grad():
        output = model(img)

        # getting the predicted value, which is the max value from the output
        pred_label = output.max(1, keepdim=True)[1]
        true_label = labels[i].view(1)

        if all_count % 1000 == 0:
            print(output)
            print(pred_label)
            print(true_label)
            print()

        all_count += 1

        # do comparison and tally if the prediction was correct, if it wasn't correct, we don't need to generate an adversarial example for it
        if true_label != pred_label:
            continue

        # getting the loss wrt the true label, we want this to be high
        # labelv = [0 if i != int(true_label) else 1 for i in range(10)]
        loss = criterion(output, true_label)

        # must zero all gradients before doing backprop
        model.zero_grad()

        # now do backprop, but wrt the image instead of the neuron weights
        loss.backward()

        # get the gradient of the image we are working with to generate the adversarial example
        img_grad = img.grad

        # getting the sign and setting a value for epsilon
        img_grad_sign = img_grad.sign()
        epsilon = 0.05

        # generate example
        adv_ex = img + epsilon * img_grad_sign

        # since the pixel values have to be between 0 and 1, we are going to clip anything that is larger than 1
        adv_ex = torch.clamp(adv_ex, 0, 1)

        # reclassify the input
        output = model(adv_ex)

        # now check to see if it was correct in misclassifying
        pred_label = torch.argmax(output)
        true_label = labels[i]

        if pred_label == true_label:
            correct_count += 1
        else:
            # successfully tricked the model, visualize what got misclassified
            if all_count % 1000 == 0:
                saved_ex.append(
                    (
                        img,
                        true_label.item(),
                        adv_ex.squeeze().detach().numpy(),
                        pred_label.item(),
                    )
                )

print("Number of Images Tested:", all_count)
print("Model Accuracy:", (correct_count / all_count))

# now visulaize the saved examples

figure = plt.figure()
n = len(saved_ex)
index = 0
for img, true_label, adv_ex, pred_label in saved_ex:
    index += 1
    plt.subplot(3, 3, index)
    plt.title("{} -> {}".format(true_label, pred_label))
    plt.imshow(adv_ex)

plt.show()