import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD #Stochastic Gradient Descent

import matplotlib.pyplot as plt
import seaborn as sns

#class initialization with inheritance

class BasicNN(nn.Module):
    def __init__(self):
        super(BasicNN, self).__init__()
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad = False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad = False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad = False)

        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad = False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad = False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad = False)

        self.final_bias = nn.Parameter(torch.tensor(-16.), requires_grad = False)

    def forward(self, inputs):  #creates a forward function that carries the value to calculate and return the output value
        inputs_to_top_relu = inputs * self.w00 + self.b00
        top_relu_output = F.relu(inputs_to_top_relu)
        scaled_top_relu_output = top_relu_output * self.w01

        inputs_to_bottom_relu = inputs * self.w10 + self.b10
        bottom_relu_output = F.relu(inputs_to_bottom_relu)
        scaled_bottom_relu_output = bottom_relu_output * self.w11

        input_to_final_relu = scaled_bottom_relu_output + scaled_top_relu_output + self.final_bias

        output = F.relu(input_to_final_relu)

        return output

class BasicNN_train(nn.Module):
    def __init__(self):
        super(BasicNN_train, self).__init__()
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad = False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad = False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad = False)

        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad = False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad = False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad = False)

        self.final_bias = nn.Parameter(torch.tensor(0.), requires_grad = True) #Please notice here! the requires_gradient is True now! which means this will need sgd

    def forward(self, inputs):  #creates a forward function that carries the value to calculate and return the output value
        inputs_to_top_relu = inputs * self.w00 + self.b00
        top_relu_output = F.relu(inputs_to_top_relu)
        scaled_top_relu_output = top_relu_output * self.w01

        inputs_to_bottom_relu = inputs * self.w10 + self.b10
        bottom_relu_output = F.relu(inputs_to_bottom_relu)
        scaled_bottom_relu_output = bottom_relu_output * self.w11

        input_to_final_relu = scaled_bottom_relu_output + scaled_top_relu_output + self.final_bias

        output = F.relu(input_to_final_relu)

        return output

#model initialization and data initialization

model = BasicNN_train()

datas = torch.tensor([0., 0.5, 1.])
labels = torch.tensor([0., 1., 0.]) #here we create the training data

#optimization stage

optimizer = SGD(model.parameters(), lr = 0.1)   #create an optimizer object with SGD, and lr is learning rate

print("Final bias, before optimization: " + str(model.final_bias.data) + "\n")

for epoch in range(100):    #an epoch means a thorough round of data training

    total_loss = 0  #stores the loss

    for iteration in range(len(datas)):

        data_i = datas[iteration]
        label_i = labels[iteration]

        output_i = model(data_i)

        loss = (output_i - label_i) ** 2    #loss has access to the derivatives in the model when we call .backward()

        loss.backward() #only accumulates the derivative information to the grad, and it won't do the optimization itself!

        total_loss += float(loss)

    if total_loss < 0.0001:
        print("Num steps: " + str(epoch))
        break

    optimizer.step()    #here is where the optimization process really starts, but not the backward(). and .step() authorize opt access to the model
    optimizer.zero_grad()   #clears the cache of the loss.backward()!!! THIS IS IMPORTANT

    print("Step: " + str(epoch) + ", loss: " + str(total_loss) + "\n")

print("Final bias, after optimization: " + str(model.final_bias.data) + "\n")




output_values = model(datas)

#here is the demonstration part

sns.set(style = 'whitegrid')

sns.lineplot(x = datas, y = output_values.detach(), color = 'green', linewidth = 2.5)   #detach() means separating the gradient value from the

plt.ylabel('Effectiveness')
plt.xlabel('Dose')

plt.show()