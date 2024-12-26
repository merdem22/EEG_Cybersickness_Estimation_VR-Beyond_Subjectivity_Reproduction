import torch.nn.functional as F
from torchvision.models._api import register_model
import torch.nn as nn
import torch
from torch.distributions import Normal


from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)

from sklearn.linear_model import (
    LogisticRegression,
    SGDClassifier,
    SGDOneClassSVM,
    Perceptron
)

from sklearn.svm import OneClassSVM, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
#from xgboost import XGBClassifier


MODELS = dict(
    OneClassSVM=OneClassSVM,
    GradBoost=GradientBoostingClassifier,
    LogisticRegression=LogisticRegression,
    DecisionTree=DecisionTreeClassifier,
    RandomForest=RandomForestClassifier,
    SVM=SVC,
    KNN=KNeighborsClassifier,
#    XGBoost=XGBClassifier,
    GaussianNB=GaussianNB,
    AdaBoost=AdaBoostClassifier,
    Perceptron=Perceptron,
    SGD=SGDClassifier,
    SGDOneClassSVM=SGDOneClassSVM,
)


@register_model('kinematic-model')
class KinematicModel(nn.Module):
    def __init__(self, n_channels=16, hidden_size=10, num_classes=1):
        super().__init__()

        n_channels = 16
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(n_channels, hidden_size, num_layers=2, batch_first=True)
        self.activation = nn.ReLU()
        self.fc1 = nn.Linear(hidden_size, hidden_size * 8)
        self.fc2 = nn.Linear(hidden_size * 8, num_classes)
        self.out = num_classes

    def forward(self, kinematic):
        self.lstm.flatten_parameters()

        x, (hn, _) = self.lstm(kinematic.permute(0, 2, 1))
        
        x = x[:, -1, :]

        x = F.relu(self.fc1(self.dropout(x)))
        x = self.fc2(self.dropout(x))

        if self.out > 1:
            return F.softmax(x, dim=1)
        else:
            return F.sigmoid(x)


#torch.jit.ScriptModule
class CNN(nn.Module):
    def __init__(self, n_channels, out_channels=10, kernel_size=4, **args):
        super(CNN, self).__init__()

        conv_kwds = dict(kernel_size=kernel_size * 2, stride=1, bias=False, padding=1)
        maxp_kwds = dict(kernel_size=kernel_size, stride=kernel_size, padding=1)

        self.conv_block1 = nn.Sequential(nn.Conv1d(n_channels, 32, **conv_kwds),
                                         nn.BatchNorm1d(32),
                                         nn.ReLU(),
                                         nn.MaxPool1d(**maxp_kwds))
        
        self.conv_block2 = nn.Sequential(nn.Conv1d(32, 64, **conv_kwds),
                                         nn.BatchNorm1d(64),
                                         nn.ReLU(),
                                         nn.MaxPool1d(**maxp_kwds))
        
        self.conv_block3 = nn.Sequential(nn.Conv1d(64, out_channels, **conv_kwds),
                                         nn.BatchNorm1d(out_channels),
                                         nn.ReLU(),
                                         nn.MaxPool1d(**maxp_kwds))
        
    #@torch.jit.script_method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        return x

class GaussianAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(GaussianAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, latent_dim * 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, input_dim),
        )

    def encode(self, x):
        z_mean, z_log_var = self.encoder(x).chunk(2, 1)
        return z_mean, z_log_var

    def decode(self, z):
        x = self.decoder(z)
        return x

    def reparameterize(self, z_mean, z_log_var):
        eps = torch.randn_like(z_mean)
        return z_mean + eps * torch.exp(z_log_var / 2.)

    def forward(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        x_recon = self.decode(z)
        return x_recon, z_mean, z_log_var
     

@register_model('multi-segment-model')
class CustomKinematicModel(nn.Module):
    def __init__(self, n_channels=16, hidden_size=10, num_classes=1):
        super().__init__()

        self.kin_encoder = CNN(16, kernel_size=2, out_channels=n_channels)
        self.kin_lstm = nn.LSTM(n_channels, hidden_size, num_layers=2, batch_first=True)

        self.eeg_encoder = CNN(24, out_channels=n_channels)
        self.eeg_lstm = nn.LSTM(n_channels, hidden_size, num_layers=2, batch_first=True)
        
        self.psd_encoder = CNN(24, out_channels=n_channels)
        self.psd_lstm = nn.LSTM(n_channels, hidden_size, num_layers=2, batch_first=True)
        
        self.fc1 = nn.Linear(3 * hidden_size, 8 * hidden_size)
        self.fc2 = nn.Linear(8 * hidden_size, 16 * hidden_size)
        self.fc3 = nn.Linear(16 * hidden_size, num_classes)
        self.out = num_classes

    def forward(self, eeg: torch.Tensor, kinematic: torch.Tensor, psd: torch.Tensor) -> torch.Tensor:
        assert eeg.shape[1] == psd.shape[1] and psd.shape[1] == kinematic.shape[1]
        self.kin_lstm.flatten_parameters()
        self.eeg_lstm.flatten_parameters()
        self.psd_lstm.flatten_parameters()

        kin_lstm_state = None
        eeg_lstm_state = None
        psd_lstm_state = None

        results = []

        for index in range(eeg.shape[1]):

            lstm_input = self.eeg_encoder(eeg[:, index, :, :,]).permute(0, 2, 1)
            lstm_output, eeg_lstm_state = self.eeg_lstm(lstm_input, hx=eeg_lstm_state)

            eeg_output = lstm_output[:, -1, :]

            lstm_input = self.psd_encoder(psd[:, index, :, :,]).permute(0, 2, 1)
            lstm_output, psd_lstm_state = self.psd_lstm(lstm_input, hx=psd_lstm_state)

            psd_output = lstm_output[:, -1, :]

            lstm_input = self.kin_encoder(kinematic[:, index, :, :]).permute(0, 2, 1)
            lstm_output, kin_lstm_state = self.kin_lstm(lstm_input, hx=kin_lstm_state)
            
            kinematic_output = lstm_output[:, -1, :]
            
            x = torch.cat([eeg_output, kinematic_output, psd_output], dim=1)
                    
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)

            if self.out > 1:
                x = F.softmax(x, dim=1)
            else:
                x = F.sigmoid(x)

            results.append(x)

        return torch.stack(results, dim=1)
    

    
@register_model('power-spectral-coeff-model')
class PSDCoeffModel(nn.Module):
    def __init__(self, n_channels=16, hidden_size=10, num_classes=1):
        super().__init__()

        n_channels = 48
        hidden_size = 12

        self.lstm = nn.LSTM(n_channels, hidden_size, num_layers=5, batch_first=False)
        
        self.fc1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.fc2 = nn.Linear(4 * hidden_size, 2 * hidden_size)
        self.fc3 = nn.Linear(2 * hidden_size, num_classes)
        self.out = num_classes

    def forward(self, linear_coeffs: torch.Tensor, quadratic_coeffs: torch.Tensor) -> torch.Tensor:

        batch = []
        for lin, quad in zip(linear_coeffs, quadratic_coeffs):
            lstm_output, _ = self.lstm(torch.cat([lin, quad], dim=1), hx=None)
            x = F.relu(self.fc1(lstm_output))
            x = F.relu(self.fc2(x))
            batch.append(self.fc3(x))
        return tuple(batch)
    


@register_model('power-spectral-difference-model')
class PSDCoeffModel(nn.Module):
    def __init__(self, n_channels=16, hidden_size=10, num_classes=1):
        super().__init__()

        self.kin_encoder = CNN(16, kernel_size=2, out_channels=n_channels)
        self.kin_lstm = nn.LSTM(n_channels, hidden_size, num_layers=2, batch_first=True)
        
        self.psd_encoder = CNN(24, out_channels=n_channels)
        self.psd_lstm = nn.LSTM(n_channels, hidden_size, num_layers=2, batch_first=True)
        
        self.fc1 = nn.Linear(2 * hidden_size, 4 * hidden_size)
        self.fc2 = nn.Linear(4 * hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.out = num_classes

    def forward(self, psd_difference: torch.Tensor, kinematic: torch.Tensor) -> torch.Tensor:
        
        lstm_input = self.psd_encoder(psd_difference).permute(0, 2, 1)
        psd_output, _ = self.psd_lstm(lstm_input, hx=None)

        lstm_input = self.kin_encoder(kinematic).permute(0, 2, 1)
        kin_output, _ = self.kin_lstm(lstm_input, hx=None)
        
        x = torch.cat([psd_output[:, -1, :], kin_output[:, -1, :]], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

@register_model('power-spectral-no-eeg-model')
class PSDCoeffModel(nn.Module):
    def __init__(self, n_channels=16, hidden_size=10, num_classes=1):
        super().__init__()

        self.kin_encoder = CNN(16, kernel_size=2, out_channels=n_channels)
        self.kin_lstm = nn.LSTM(n_channels, hidden_size, num_layers=2, batch_first=True)
        
        self.fc1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.fc2 = nn.Linear(4 * hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.out = num_classes

    def forward(self, kinematic: torch.Tensor) -> torch.Tensor:

        lstm_input = self.kin_encoder(kinematic).permute(0, 2, 1)
        kin_output, _ = self.kin_lstm(lstm_input, hx=None)
        
        x = kin_output[:, -1, :]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

@register_model('power-spectral-no-kinematic-model')
class PSDCoeffModel(nn.Module):
    def __init__(self, n_channels=16, hidden_size=10, num_classes=1):
        super().__init__()
        
        self.psd_encoder = CNN(24, out_channels=n_channels)
        self.psd_lstm = nn.LSTM(n_channels, hidden_size, num_layers=2, batch_first=True)
        
        self.fc1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.fc2 = nn.Linear(4 * hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.out = num_classes

    def forward(self, psd_difference: torch.Tensor) -> torch.Tensor:
        
        lstm_input = self.psd_encoder(psd_difference).permute(0, 2, 1)
        psd_output, _ = self.psd_lstm(lstm_input, hx=None)
        
        x = psd_output[:, -1, :]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x