import torch
import torch.nn as nn
from models.functions import ReverseLayerF


class FCModel(nn.Module):

    def __init__(self):
        super(FCModel, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_fc1', nn.Linear(15, 100))
        self.feature.add_module('f_bn1', nn.BatchNorm1d(100))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.skip_connection = nn.Identity()
        self.feature.add_module('f_fc2', nn.Linear(100, 100))
        self.feature.add_module('f_bn2', nn.BatchNorm1d(100))
        self.feature.add_module('f_relu2', nn.ReLU(True))
        self.feature.add_module('f_fc3', nn.Linear(100, 100))
        self.feature.add_module('f_bn3', nn.BatchNorm1d(100))
        self.feature.add_module('f_relu3', nn.ReLU(True))
        self.process = nn.Sequential()
        self.process.add_module('f_fc4', nn.Linear(100, 100))
        self.process.add_module('f_bn4', nn.BatchNorm1d(100))
        self.process.add_module('f_relu4', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(115, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 3))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(115, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))


    def forward(self, input_data, alpha):
        feature = self.feature(input_data)
        process = self.process(feature)
        process_data = torch.cat([process, input_data], dim=1)
        reverse_feature = ReverseLayerF.apply(process_data, alpha)
        class_output = self.class_classifier(process_data)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output