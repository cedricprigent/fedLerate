import torch
from torch import nn
import torch.nn.functional as F

from torchvision.models import resnet18, ResNet18_Weights


class LeNet5(nn.Module):

    def __init__(self, input_h=28, in_channels=1, num_classes=10):
        super(LeNet5, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.features = LeNet5_Features(input_h=input_h, in_channels=in_channels)
        self.classifier = LeNet5_Classifier(features_output_size=self.features.features_output_size, in_channels=in_channels, num_classes=num_classes)

    def forward(self, x):
        x = self.features(x)
        probas = self.classifier(x)
        return probas


class PFL_LeNet5(nn.Module):

    def __init__(self, input_h=28, in_channels=1, num_classes=10):
        super(PFL_LeNet5, self).__init__()

        self.global_layers = LeNet5_Features(input_h=input_h, in_channels=in_channels)
        self.local_layers = LeNet5_Classifier(features_output_size=self.global_layers.features_output_size, in_channels=in_channels, num_classes=num_classes)

    def forward(self, x):
        x = self.global_layers(x)
        probas = self.local_layers(x)
        return probas


class LeNet5_Features(nn.Module):

    def __init__(self, input_h=28, in_channels=1):
        super(LeNet5_Features, self).__init__()
        
        # conv_maxpool_output = (input_h - kernel_size + 1) / 2 
        features_output_h = int((((input_h - 4)/2) - 4)/2)
        features_output_size = features_output_h * features_output_h

        self.in_channels = in_channels
        self.features_output_size = features_output_size

        self.features = nn.Sequential(
            
            nn.Conv2d(in_channels, 16*in_channels, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16*in_channels, 32*in_channels, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x

    def set_weights(self, weights):
        """Set model weights from a list of NumPy ndarrays."""
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.load_state_dict(state_dict, strict=True)


class LeNet5_Classifier(nn.Module):

    def __init__(self, features_output_size, in_channels=1, num_classes=10):
        super(LeNet5_Classifier, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.Linear(features_output_size*32*in_channels, 120*in_channels),
            nn.Tanh(),
            nn.Linear(120*in_channels, 84*in_channels),
            nn.Tanh(),
            nn.Linear(84*in_channels, num_classes),
        )

    def forward(self, x):
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)
        return probas

    def set_weights(self, weights):
        """Set model weights from a list of NumPy ndarrays."""
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.load_state_dict(state_dict, strict=True)


class DummyModel(nn.Module):
    def __init__(self, input_size=1, num_classes=1):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return probas


class PFL_ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        self.global_layers = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3
        )
        self.local_layers = nn.Sequential(
            model.layer4,
            model.avgpool,
            nn.Flatten(),
            torch.nn.Linear(model.fc.in_features, 7)
        )

    def forward(self, x):
        x = self.global_layers(x)
        x = self.local_layers(x)
        return x

# class Encoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 32, bias=False), # No bias!
#             nn.Tanh() # This squashes the output into [-1, 1], keeping it near the origin
#         )

#     def forward(self, x):
#         return self.net(x)
class TransformerEncoder(nn.Module):
    def __init__(self, beat_len=64, input_channels=1, d_model=128, nhead=4, num_layers=2, num_classes=5):
        super(TransformerEncoder, self).__init__()
        self.beat_len = beat_len
        self.d_model = d_model
        
        # 1. Input Projection: Projects signal to latent dimension
        # Using kernel_size=3 keeps the model sensitive to local waveform features
        self.embedding = nn.Conv1d(input_channels, d_model, kernel_size=3, padding=1)
        
        # 2. Positional Encoding: Learned parameters
        self.pos_encoder = nn.Parameter(torch.randn(1, d_model, beat_len))
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Final Classification Head
        # Dynamically calculate the flattened dimension based on beat_len
        self.flatten_dim = d_model * beat_len
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Input x shape: (Batch, Channel, BEAT_LEN) -> (Batch, 1, 64)
        
        if x.dim() == 2:
            x = x.unsqueeze(1) 
        
        # Now that it's 3D, we can safely slice if needed
        # If x has more than 1 channel, take the first one
        if x.shape[1] > 1:
            x = x[:, 0:1, :]
            
        # Project and add position
        x = self.embedding(x) + self.pos_encoder 
        
        # Transformer expects (Batch, Seq_Len, d_model)
        x = x.permute(0, 2, 1) 
        
        # Pass through Transformer
        x = self.transformer(x)
        
        # Flatten: (Batch, d_model * BEAT_LEN)
        x = x.reshape(x.size(0), -1)
        
        # Classifier
        return self.fc(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )

    def forward(self, z):
        return self.net(z)


class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
class PFL_AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.global_layers = encoder
        self.local_layers = decoder

    def forward(self, x):
        z = self.global_layers(x)
        return self.local_layers(z)

class DeepSVDD(nn.Module):
    def __init__(self, encoder, center):
        super().__init__()
        self.encoder = encoder
        # The center is a fixed point in the latent space
        self.register_buffer("center", center)

    def forward(self, x):
        return self.encoder(x)


def load_model(model_name, device):

    if model_name == 'DummyModel':
        model = DummyModel().to(device)
    elif model_name == 'LeNet5':
        model = LeNet5().to(device)
    elif model_name == 'LeNet5_Features':
        model = LeNet5_Features().to(device)
    elif model_name == 'LeNet5_Classifier':
        model = LeNet5_Classifier().to(device)
    elif model_name == 'PFL_LeNet5':
        model = PFL_LeNet5().to(device)
    elif model_name == 'PFL_LeNet5_3C':
        model = PFL_LeNet5(in_channels=3).to(device)
    elif model_name == 'ResNet18':
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
        model.fc = torch.nn.Linear(model.fc.in_features, 7).to(device)
    elif model_name == 'PFL_ResNet18':
        model = PFL_ResNet18().to(device)
    elif model_name == 'AutoEncoder':
        encoder = Encoder().to(device)
        decoder = Decoder().to(device)
        model = AutoEncoder(encoder, decoder).to(device)
    elif model_name == 'PFL_AutoEncoder':
        encoder = Encoder().to(device)
        decoder = Decoder().to(device)
        model = PFL_AutoEncoder(encoder, decoder).to(device)
    elif model_name == 'TransformerEncoder':
        model = TransformerEncoder().to(device)
    elif model_name == 'DeepSVDD':
        encoder = Encoder().to(device)
        model = DeepSVDD(encoder, torch.zeros(32)).to(device)
    else:
        raise NotImplementedError

    return model


def get_global_layers(state_dict, n_local_layers):
    n_layers = len(state_dict)
    return {
        k: v
        for i, (k, v) in enumerate(state_dict.items())
        if n_layers - i > n_local_layers * 2
    }

def get_local_layers(state_dict, n_local_layers):
    n_layers = len(state_dict)
    return {
        k: v
        for i, (k, v) in enumerate(state_dict.items())
        if n_layers - i <= n_local_layers * 2
    }