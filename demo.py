from DeepLin import DeepLing
import torch.nn as nn
import numpy as np


training_data = np.random.random((1000, 10))
training_label = np.random.randint(0, 2, (1000, 1))
test_data = np.random.random((500, 10))
test_label = np.random.randint(0, 2, (500, 1))


class DemoNet(nn.Module):
    def __init__(self):
        super(DemoNet, self).__init__()
        self.modelName = "DemoNet"
        self.fc = nn.Linear(in_features=10, out_features=2)

    def forward(self, x):
        return self.fc(x)


model = DeepLing(DemoNet, 'CEP', optFunc='Adam', epoch=100, early_stop=10)

print("训练")
model.getTrainingDataLoader(training_data, training_label)
model.train()

print('测试')
model.getTestDataLoader(test_data, test_label)
test_pred = model.test()

model.score(test_label, test_pred)





