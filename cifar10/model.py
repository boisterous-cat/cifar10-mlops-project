import torch.nn as nn
import torch.nn.functional as F


class MyResNet(nn.Module):
    """
    MyResNet is a convolutional neural network model designed for image classification tasks.
    It consists of several convolutional layers followed by max pooling, dropout, and fully connected layers.

    Attributes:
        conv1 (nn.Conv2d): The first convolutional layer that takes 3 input channels (RGB images)
                           and outputs 32 channels.
        conv2 (nn.Conv2d): The second convolutional layer that takes 32 input channels and outputs 64 channels.
        conv3 (nn.Conv2d): The third convolutional layer that takes 64 input channels and outputs 128 channels.
        pool (nn.MaxPool2d): Max pooling layer to downsample the feature maps.
        dropout (nn.Dropout): Dropout layer to reduce overfitting by randomly setting a fraction of input units to zero.
        fc1 (nn.Linear): The first fully connected layer that takes the flattened input and outputs 256 features.
        fc2 (nn.Linear): The second fully connected layer that takes 256 input features and outputs 128 features.
        fc3 (nn.Linear): The final fully connected layer that maps the last hidden layer to the number of classes.

    Parameters:
        num_classes (int): The number of output classes for classification. Default is 10.

    Example:
        model = MyResNet(num_classes=10)
        output = model(torch.randn(1, 3, 32, 32))  # Example input for a 32x32 RGB image
    """

    def __init__(self, num_classes=10):
        super(MyResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(
            128 * 4 * 4, 256
        )  # After pooling, the size is reduced to 4x4
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W), where N is the batch size,
                              C is the number of channels, H is the height, and W is the width of the input image.

        Returns:
            torch.Tensor: Output tensor of shape (N, num_classes) representing the class scores for each input.
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 128 * 4 * 4)  # Flatten the tensor for fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Final output

        return x
