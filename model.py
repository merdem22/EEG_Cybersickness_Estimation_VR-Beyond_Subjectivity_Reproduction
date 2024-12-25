import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader


# Cognitive Representation Learning (Stage 1)
class CognitiveRepresentationNet(nn.Module):
    def __init__(self, n1: int = 14, n2: int = 7, m1: int = 8, m2: int = 4):
        super(CognitiveRepresentationNet, self).__init__()

        # Temporal network (1 x m kernels)
        num_cognitive_features = 936
        num_hidden_neurons = 128
        num_sickness_level_logits = 5
        # Kernel sizes decrease by half: (1x28), (1x14), (1x7)
        self.temporal_conv1 = nn.Conv2d(8, 16, kernel_size=(1, n1), stride=1, padding=0)
        self.temporal_bn1 = nn.BatchNorm2d(16)
        self.temporal_conv2 = nn.Conv2d(
            16, 16, kernel_size=(1, n1), stride=(1, 2), padding=0
        )
        self.temporal_bn2 = nn.BatchNorm2d(16)
        self.temporal_conv3 = nn.Conv2d(
            16, 8, kernel_size=(1, n2), stride=(1, 2), padding=0
        )
        self.temporal_bn3 = nn.BatchNorm2d(8)
        self.temporal_maxpool = nn.AdaptiveMaxPool1d(468)  # Global max pooling

        # Spectral network (n x 1 kernels)
        # Kernel sizes decrease by half: (8x1), (4x1), (2x1)
        self.spectral_conv1 = nn.Conv2d(8, 16, kernel_size=(m1, 1), stride=1, padding=0)
        self.spectral_bn1 = nn.BatchNorm2d(16)
        self.spectral_conv2 = nn.Conv2d(
            16, 16, kernel_size=(m1, 1), stride=(2, 1), padding=0
        )
        self.spectral_bn2 = nn.BatchNorm2d(16)
        self.spectral_conv3 = nn.Conv2d(
            16, 8, kernel_size=(m2, 1), stride=(2, 1), padding=0
        )
        self.spectral_bn3 = nn.BatchNorm2d(8)
        self.spectral_maxpool = nn.AdaptiveMaxPool1d(468)  # Global max pooling

        # Fully connected layers for classification
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.fcl = nn.Sequential(
            nn.Linear(num_cognitive_features, num_hidden_neurons),
            nn.LeakyReLU(0.1),
            nn.Linear(
                num_hidden_neurons, num_sickness_level_logits
            ),  # Assuming 5 cybersickness levels
        )

    def forward(self, x):
        # x shape: (batch_size, 8, 64, 53)

        # Temporal Path
        x_temp = self.leaky_relu(self.temporal_bn1(self.temporal_conv1(x)))
        x_temp = self.leaky_relu(self.temporal_bn2(self.temporal_conv2(x_temp)))
        x_temp = self.leaky_relu(self.temporal_bn3(self.temporal_conv3(x_temp)))
        x_temp = self.temporal_maxpool(x_temp.flatten(1))  # Flatten

        # Spectral Path
        x_spec = self.leaky_relu(self.spectral_bn1(self.spectral_conv1(x)))
        x_spec = self.leaky_relu(self.spectral_bn2(self.spectral_conv2(x_spec)))
        x_spec = self.leaky_relu(self.spectral_bn3(self.spectral_conv3(x_spec)))
        x_spec = self.spectral_maxpool(x_spec.flatten(1))  # Flatten

        # Concatenate Temporal and Spectral Features
        cognitive_repr = torch.cat((x_temp, x_spec), dim=1)

        # Fully connected layers
        sickness_level = self.fcl(cognitive_repr)

        return sickness_level, cognitive_repr  # Output cybersickness levels


# Cybersickness Learning (Stage 2)
class CybersicknessLearningNet(nn.Module):
    def __init__(self):
        super(CybersicknessLearningNet, self).__init__()

        # Video Encoder (ResNet18 + LSTM)
        self.resnet = models.resnet18(weights="DEFAULT")
        self.resnet.fc = nn.Identity()  # Remove last fully connected layer

        num_visual_features = 128
        num_cognitive_features = 936

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=num_visual_features,
            num_layers=2,
            batch_first=True,
        )

        # Fully connected layers for final cybersickness prediction
        hidden_layer_size = int((num_visual_features * num_cognitive_features) ** 0.5)
        self.fcl_cognitive = nn.Sequential(
            nn.Linear(num_visual_features, hidden_layer_size),
            nn.LeakyReLU(0.1),
            nn.Linear(
                hidden_layer_size, num_cognitive_features
            ),  # Output cybersickness levels
        )

        hidden_layer_size = int(
            ((num_visual_features + num_cognitive_features) * 5) ** 0.5
        )
        self.fcl_sickness = nn.Sequential(
            nn.Linear(num_cognitive_features + num_visual_features, hidden_layer_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_layer_size, 5),  # Output cybersickness levels
        )

    def forward(self, x_video):
        # ResNet18 for video frame processing
        batch_size, seq_len, c, h, w = x_video.size()
        x_video = x_video.view(batch_size * seq_len, c, h, w)
        x_video = self.resnet(x_video)  # Extract spatial features

        # LSTM for temporal processing
        x_video = x_video.view(batch_size, seq_len, -1)
        _, (hn, _) = self.lstm(x_video)

        # Combine visual features with cognitive representation
        visual_features = hn[-1]  # Take the final hidden state from LSTM
        cognitive_features = self.fcl_cognitive(visual_features)

        x_combined = torch.cat((cognitive_features, visual_features), dim=1)

        cybersickness_level = self.fcl_sickness(x_combined)

        return cybersickness_level, cognitive_features


# Combine the two stages
class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        self.cognitive_net = CognitiveRepresentationNet()
        self.cybersickness_net = CybersicknessLearningNet()

    def forward(self, x_eeg, x_video):
        eeg_cybersickness_level, cognitive_representation = self.cognitive_net(x_eeg)
        vid_cybersickness_level, cognitive_features = self.cybersickness_net(x_video)
        return (
            eeg_cybersickness_level,
            cognitive_representation,
            vid_cybersickness_level,
            cognitive_features,
        )


if __name__ == "__main__":
    # Example usage
    model = CombinedModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Example inputs
    from loader import video_to_numpy_moviepy

    video_path = "/home/adhd/data/juliette-eeg/footage/Take03/side.mp4"  # Replace with your actual video path
    video_data = video_to_numpy_moviepy(video_path)
    video_data = torch.from_numpy(video_data) / 255.0

    eeg_data = torch.randn(10, 8, 64, 53)  # (batch_size, channels, height, width)
    # video_data = torch.randn(
    #     10, 30, 3, 224, 224
    # )  # (batch_size, seq_len, channels, height, width)

    # Forward pass
    cog_repr, sick_level = model(eeg_data, video_data)
    print(cog_repr.shape, sick_level)
