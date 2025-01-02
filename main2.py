# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

train_set = datasets.MNIST(root="data", download=True, transform=ToTensor())
test_set = datasets.MNIST(root="data",
                          download=True,
                          transform=ToTensor(),
                          train=False)
"""
As described above, the DataLoader class specifies how to load the data. Here
we have specified a batch size of 64. Usually, for
the train set the shuffle parameter is set to True, meaning that the dataloader
iterates through the dataset in random order each time. We won't shuffle the
train set in this exercise because we want to train multiple models under
similar conditions and compare the results. The test set isn't
shuffled, because that way we can easily access the same samples which eases the
comparison of models and their predictions.
"""
train_loader = DataLoader(train_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

"""# Models

Next we will take a look at how we can specify the architecture of our neural network (in deep learning our model almost always will be a neural network). The base class of all neural networks in PyTorch is the `torch.nn.Module` class. Every component of a neural network inherits from this class and the final neural network as well. Next is an example of how one could write an MLP (Multi-layer Perceptron a.k.a. Fully connected neural network) of fixed size in PyTorch. A class inheriting from `nn.Module` has to specify 2 functions:
- `__init__()`: here we define all components and parameters of the model, as well as any configuration variables that may be used to alter how the network works.
- `forward()`: here specify the forward pass by making use of the components that we have instantiated in the `__init__()` function.
"""

from torch import nn


class SimpleFixedMLP(nn.Module):

    def __init__(self):


        super().__init__()
        self.flatten = nn.Flatten()
        self.input_layer = nn.Linear(in_features=28 * 28,
                                     out_features=32,
                                     bias=True)
        self.hidden_layer1 = nn.Linear(in_features=32,
                                       out_features=32,
                                       bias=True)
        self.hidden_layer2 = nn.Linear(in_features=32,
                                       out_features=32,
                                       bias=True)
        self.output_layer = nn.Linear(in_features=32,
                                      out_features=10,
                                      bias=True)
        self.activation = nn.ReLU()

    def forward(self, input):
        """
        This is the forward pass. It takes as an argument the input to the model
        and returns the input after applying the components.
        """
        out = self.flatten(input)
        out = self.input_layer(out)
        out = self.activation(out)
        out = self.hidden_layer1(out)
        out = self.activation(out)
        out = self.hidden_layer2(out)
        out = self.activation(out)
        out = self.output_layer(out)
        return out

"""## Task 1a (20 P)
Write another class `SimpleMLP`, that takes as an argument a list of dimensionalities of the layers, such that we can instantiate MLPs of arbitrary size using that class. If we'd like to replicate the above SimpleFixedMLP we could set the `dims` parameter to `[28*28, 32, 32, 32, 10]`. The activation function should remain the ReLU function. Also don't be confused by how the solution may look like, you don't have to put
all the layers into a single variable.

It's very important for all the variables defined in your `__init__()` function to be either a subclass of `nn.Module` (most members of the `torch.nn` package inherit from `torch.nn.Module`) or to be wrapped with `nn.Parameter(...)`. This is because otherwise, when training our model, these variables may not be considered "learnable" and won't change. Furthermore, when saving the model to a file, PyTorch only saves variables that it recognizes as parameters. These resources may help: [a blog post](https://blog.paperspace.com/pytorch-101-advanced/), [stackoverflow question](https://stackoverflow.com/questions/50935345/understanding-torch-nn-parameter) or [pytorch nn.Parameter docs](https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html)  

Hint: you may use the [`ModuleList`](https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html) module.
"""

class SimpleMLP(nn.Module):

    def __init__(self, dims: list[int]):
        super().__init__()



        self.modulelist = nn.ModuleList([nn.Flatten()])
        for in_features, out_features in zip(dims[:-2], dims[1:]):
            self.modulelist.extend([nn.Linear(in_features, out_features), nn.ReLU()])
        self.modulelist.append(nn.Linear(dims[-2], dims[-1]))

    def forward(self, input: torch.Tensor):


      out = input
      for l in self.modulelist:
          out = l(out)
      return out

"""## Task 1b) (10 P)

Usually, if we have such simple architectures as an MLP, where the input is just sequentially transformed, its not necessary to define a new class. Instead we can just use the [`nn.Sequential`](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html) module, which takes a list of modules as an input and applies them sequentially. In this task you should instantiate 3 separate models, which all will have the same structure. The first model should use the SimpleFixedMLP class, the second model should use the SimpleMLP class, with the layer dimensions matching the ones of the SimpleFixedMLP class, and one model using the nn.Sequential module, again with the same dimensionalities as the other 2 models.
"""

model1 =  SimpleFixedMLP()
model2 = SimpleMLP(dims=[28 * 28, 32, 32, 32, 10])
model3 = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 32), nn.ReLU(),
                         nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32),
                         nn.ReLU(), nn.Linear(32, 10))

"""The three models you just created might have been created in different ways, but internally they should all work the same way. Each layer is consists of a matrix-vector multiplication followed by a ReLU nonlinearity, except for the last layer. The only thing that differs is the initialization of parameters, which happens randomly. The following code should demonstrate that. For each model we will print the shapes of the weight matrices and biases along with the first 3 entries of the first weight matrix."""

for i, model in enumerate([model1, model2, model3]):
    print(f"Model {i+1} parameter shapes:")
    for i, param in enumerate(model.parameters()):
        print(f"\t{param.shape} {param.detach()[0, :3] if i == 0 else ''}")

"""## Task 2a) (20 P)

Implement a function for the above described training loop. The loop over multiple epochs (1 epoch = 1 iteration through the whole dataset) and through the dataset are given, you just have to add the code for a single iteration. To inspect the training process in the next task, add the loss for each iteration to the `losses` array.

You might have noticed that while our model outputs a 10-dimensional vector of values in $]-∞, ∞[$ the labels are scalars indicating the corresponding class, e.g. `torch.Tensor(3.)` for the digit 3. More on that will be covered in the upcoming homework, for now its enough to know that you can feed them as is into the loss function `loss_fn` like so `loss_fn(predictions, labels)`

You are free to add print statements as you like. For example you could print the loss every few iterations so that we can track the progress when training the models in the next task. This is not part of the task though.
"""

from typing import Callable


def train(model: nn.Module,
          dataloader: DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
          n_epochs: int = 20):
    for epoch in range(n_epochs):
        losses = []
        for batch, labels in dataloader:
            # --------------------------------------------------
            optimizer.zero_grad()
            preds = model(batch)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
            # --------------------------------------------------
            #losses.append(
             #   solution_2a_train(model, batch, labels, optimizer, loss_fn))
            # --------------------------------------------------
    return losses

"""## Task 2b) (10 P)

Next we want to actually train models. We have already defined our dataloader objects above in the "Dataset (MNIST)" cell. To train models, you are given a new array of 6 models with less parameters to ensure executability on every machine. Your task will be to define 6 corresponding optimizers, one for each model, e.g. `optimizers[i]` corresponds to `models[i]`. We will use `torch.optim.Adam()` with a learning rate of 0.04 for this task. As our loss function we will use the `nn.CrossEntropyLoss()` (more on that in the "Tasks / Loss functions" lecture).

Then use the `train()` function created in task 2a) to train our models. For your final submission please keep the number of epochs at 1, though feel free to play around with the number of epochs, and also the batch size (specified in the dataloader at the top of this notebook). Make sure you use the correct dataloader.

After having trained all models, plot their losses in a single plot to compare their convergence behaviour. Due to high fluctuations we have smoothed the losses array using a moving average with a window size of 25.
"""

import numpy as np
from torch.optim import Adam
import matplotlib.pyplot as plt

loss_fn = nn.CrossEntropyLoss()

models = [SimpleMLP(dims=[28 * 28, 32, 10]) for _ in range(10)]


optimizers = [Adam(model.parameters(), lr=0.04) for model in models]

assert len(models) == len(optimizers)

    #train_loader: DataLoader = get_global_var_from_notebook("train_loader")

for i, (model, optimizer) in enumerate(zip(models, optimizers)):
    print(f"Training model {i+1} ...")

    losses = train(model, train_loader, optimizer, loss_fn, n_epochs = 3)

    filter_size = 25
    losses_np = np.array(losses)
    smoothing_filter = np.ones(filter_size) / filter_size

    losses_smooth = np.convolve(losses_np, smoothing_filter, mode="valid")

    plt.plot(losses_smooth, label=f"Model {i+1}")
plt.legend(loc="upper right")
plt.show()
#solution_2b_train(train, models, optimizers, loss_fn, n_epochs=1)
# --------------------------------------------------

"""# Evaluation

## Task 3a) (20 P)

In this task we will compare the overall accuracy of our models. The accuracy is defined as follows:

\begin{equation} \text{accuracy } = \frac{\text{\# correctly classified samples}}{\text{\# samples}} \end{equation}

Complete the missing part of the following function that gets as input a model and a dataloader and outputs the accuracy metric.
"""

def compute_accuracy(model: nn.Module, dataloader: DataLoader) -> float:
    "Compute accuracy of `model` on the dataset in `dataloader`."
    n_correctly_classified = 0

    for batch, labels in dataloader:

        # --------------------------------------------------
        logits = model(batch)
        preds = torch.argmax(logits, dim=-1)
        n_correctly_classified += torch.sum(preds == labels).item()

        # --------------------------------------------------
        #n_correctly_classified += solution_3a_compute_accuracy(
         #   model, batch, labels)
        # --------------------------------------------------

    accuracy = n_correctly_classified / len(dataloader.dataset)
    return accuracy

test_accuracies = [compute_accuracy(model, test_loader) for model in models]

for i, test_acc in enumerate(test_accuracies):
    print(f"Model {i+1} Test accuracy: {test_acc:.4f}")

"""## Task 3b) (20 P)

Next we will take a look at the accuracy the models achieve per class. Write a function, similar to the one in the previous task, that computes the accuracy for a model on a dataset, for each class in the MNIST dataset. The output of that function should be a Python dictionary, that has as keys the integers 0-9 (classes of the MNIST dataset), and as values the corresponding accuracy. The per class accuracy is computed as follows:

\begin{equation} \text{accuracy}_i = \frac{\text{\# correctly classified samples of class } i}{\text{\# samples of class } i} \end{equation}
"""

def compute_accuracy_per_class(model: nn.Module,
                               dataloader: DataLoader) -> dict[int, float]:
    "Compute the accuracy of `model` for each class of the dataset in `dataloader`."

    # --------------------------------------------------
    n_correctly_classified_per_class = {k: 0 for k in range(10)}
    n_samples_per_class = {k: 0 for k in range(10)}
    for batch, labels in dataloader:
        logits = model(batch)
        preds = torch.argmax(logits, dim=-1)
        correctly_classified = labels == preds

        for cls, correct in zip(labels.flatten().tolist(),
                                correctly_classified.flatten().tolist()):
            n_samples_per_class[cls] += 1
            if correct:
                n_correctly_classified_per_class[cls] += 1

    accuracy_per_class = {
        k: n_correctly_classified_per_class[k] / n_samples_per_class[k]
        for k in range(10)
    }
    return accuracy_per_class
    # --------------------------------------------------
    #return solution_3b_accuracy_per_class(model, dataloader)
    # --------------------------------------------------

"""## Conclusion

Next, there is a function which will plot the accuracy of each model in a bar chart. The x-axis is labelled with the classes, e.g. digits 0-9, and the y-axis corresponds to the accuracy of a given model on that class.

Remember, all 6 models have exactly the same number of parameters, they all have been trained for exactly one epoch on the same ordering of dataset samples, and they have been optimized using the exact same optimizer. Yet, we see significantly different results in terms of class-specific accuracy.

This is due to the highly non-convex loss function, and the random initialization. This causes the models to land in different local optima of the loss function which in turn causes them to perform differently.
"""

import matplotlib.pyplot as plt


def accuracies_bar_plot(per_class_accuracies, label):
    labels = list(range(10))

    bar_heights = {
        f"Model {i+1}": list(per_class_accuracies[i].values())
        for i, model in enumerate(models)
    }

    width = 0.09
    multiplier = -3

    for model_id, accuracies in bar_heights.items():
        offset = width * multiplier
        rects = plt.bar([l + offset for l in labels],
                        accuracies,
                        width,
                        label=model_id)
        multiplier += 1

    plt.ylabel(label)
    plt.xlabel("MNIST digit")
    plt.xticks([l + width for l in labels], labels)
    plt.legend(loc="upper right", ncols=2)
    plt.show()


test_per_class_accuracies = [
    compute_accuracy_per_class(model, test_loader) for model in models
]

accuracies_bar_plot(test_per_class_accuracies, "Test Accuracy")
