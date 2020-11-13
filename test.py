import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision as tv

# loading the model
model = torch.load("model.pth")

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

valset = tv.datasets.MNIST(
    "~/Documents/Grad/AMLProject", download=False, train=False, transform=transform
)

valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

criterion = torch.nn.CrossEntropyLoss()

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

        """if all_count % 1000 == 0:
            print(output)
            print(pred_label)
            print(true_label)
            print()"""

        all_count += 1

        # do comparison and tally if the prediction was correct, if it wasn't correct, we don't need to generate an adversarial example for it
        if true_label != pred_label:
            continue

        # getting the loss wrt the true label, as an adversary we want this to be high
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
            if len(saved_ex) < 6:
                saved_ex.append(
                    (
                        img.squeeze().detach().numpy(),
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
    plt.subplot(3, 4, index)
    plt.title("Actual: {}".format(true_label))
    plt.axis("off")
    plt.imshow(adv_ex.reshape(28, 28))
    index += 1
    plt.subplot(3, 4, index)
    plt.title("Predicted: {}".format(pred_label))
    plt.axis("off")
    plt.imshow(img.reshape(28, 28))

plt.show()