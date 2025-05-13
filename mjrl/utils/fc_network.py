import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class RNNNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim,
                 hidden_sizes=(256,256),
                lstm_hidden_size=64,  # Number of hidden units in the LSTM
                 mlp_hidden_size=64, 
                 nonlinearity='tanh',  # Activation after RNN
                 in_shift=None,
                 in_scale=None,
                 out_shift=None,
                 out_scale=None,
                 log_dir='runs/rnn_activations'):
        super(RNNNetwork, self).__init__()
        self.i = 0
        self.batch_size = 1
        self.hidden_state = None
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        assert type(hidden_sizes) == tuple
        self.layer_sizes = (obs_dim,) + hidden_sizes + (act_dim,)
        self.set_transformations(in_shift, in_scale, out_shift, out_scale)
        self.lstm = nn.LSTM(input_size=obs_dim, hidden_size=lstm_hidden_size, batch_first=True)

        self.fc = nn.Linear(lstm_hidden_size, mlp_hidden_size)
        self.output_layer = nn.Linear(mlp_hidden_size, act_dim)
        
        if nonlinearity == 'relu':
            self.nonlinearity = torch.relu 
        elif nonlinearity == 'tanh':
            self.nonlinearity = torch.tanh
        elif nonlinearity == 'sigmoid':
            self.nonlinearity = torch.sigmoid
        elif nonlinearity == 'softplus':
            self.nonlinearity = nn.functional.softplus
        else:
            raise ValueError("Nonlinearity must be 'tanh', 'relu', or 'sigmoid'")

        self.activations = []
        self.writer = SummaryWriter(log_dir)
        
        
    def initialize_hidden_state(self, batch_size, device):
        return (torch.zeros(1, batch_size, 256).to(device),
                torch.zeros(1, batch_size, 256).to(device))

    def set_transformations(self, in_shift=None, in_scale=None, out_shift=None, out_scale=None):
        self.transformations = dict(in_shift=in_shift,
                                    in_scale=in_scale,
                                    out_shift=out_shift,
                                    out_scale=out_scale)
        self.in_shift = torch.from_numpy(np.float32(in_shift)) if in_shift is not None else torch.zeros(self.obs_dim)
        self.in_scale = torch.from_numpy(np.float32(in_scale)) if in_scale is not None else torch.ones(self.obs_dim)
        self.out_shift = torch.from_numpy(np.float32(out_shift)) if out_shift is not None else torch.zeros(self.act_dim)
        self.out_scale = torch.from_numpy(np.float32(out_scale)) if out_scale is not None else torch.ones(self.act_dim)

    def forward(self, x, hidden_state=None):     
        self.activations = []
        
        if self.hidden_state == None:
            self.hidden_state = self.initialize_hidden_state(self.batch_size, x.device)
        
        if x.size(0) != 1 or x.size(0) != self.batch_size:
            self.hidden_state = self.initialize_hidden_state(x.size(0), x.device)

        self.batch_size = x.size(0)

        # Normalize inputs
        x = (x - self.in_shift) / (self.in_scale + 1e-8)

        # Ensure x has a batch and sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add a sequence dimension
                   
        # Pass through LSTM layer
        lstm_out, self.hidden_state= self.lstm(x, self.hidden_state)

        lstm_out = self.nonlinearity(lstm_out)
        
        self.activations.append(lstm_out)
        
        out = lstm_out[:, -1, :]  # Use the output from the last time step
        
        # Pass through MLP layer
        out = self.nonlinearity(self.fc(out))
        self.activations.append(out)
        out = self.output_layer(out)

        # Scale and shift output
        out = out * self.out_scale + self.out_shift
        
        return out, hidden_state

    def close_writer(self):
        self.writer.close()

    def to(self, device):
        self.device = device
        self.in_shift = self.in_shift.to(device)
        self.in_scale = self.in_scale.to(device)
        self.out_shift = self.out_shift.to(device)
        self.out_scale = self.out_scale.to(device)
        super().to(device)
        
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the SummaryWriter from the state
        if 'writer' in state:
            del state['writer']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Recreate the SummaryWriter (won't be in pickled state)
        self.writer = SummaryWriter()
        
class RecurrentNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim,
                 hidden_sizes=(256,256),
                 rnn_hidden_size=64,  # Number of hidden units in the LSTM
                 mlp_hidden_size=64, 
                 nonlinearity='tanh',  # Activation after RNN
                 in_shift=None,
                 in_scale=None,
                 out_shift=None,
                 out_scale=None,
                 log_dir='runs/rnn_activations'):
        super(RecurrentNetwork, self).__init__()
        self.i = 0
        self.batch_size = 1
        self.hidden_state = None
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        assert type(hidden_sizes) == tuple
        self.layer_sizes = (obs_dim,) + hidden_sizes + (act_dim,)
        self.set_transformations(in_shift, in_scale, out_shift, out_scale)
        self.rnn = nn.RNN(input_size=obs_dim, hidden_size=rnn_hidden_size, batch_first=True)

        self.fc = nn.Linear(rnn_hidden_size, mlp_hidden_size)
        self.output_layer = nn.Linear(mlp_hidden_size, act_dim)
        
        if nonlinearity == 'relu':
            self.nonlinearity = torch.relu 
        elif nonlinearity == 'tanh':
            self.nonlinearity = torch.tanh
        elif nonlinearity == 'sigmoid':
            self.nonlinearity = torch.sigmoid
        elif nonlinearity == 'softplus':
            self.nonlinearity = nn.functional.softplus
        else:
            raise ValueError("Nonlinearity must be 'tanh', 'relu', or 'sigmoid'")

        self.activations = []
        self.writer = SummaryWriter(log_dir)
        
        
    def initialize_hidden_state(self, batch_size, device):
        return torch.zeros(1, batch_size, self.rnn.hidden_size).to(device)

    def set_transformations(self, in_shift=None, in_scale=None, out_shift=None, out_scale=None):
        self.transformations = dict(in_shift=in_shift,
                                    in_scale=in_scale,
                                    out_shift=out_shift,
                                    out_scale=out_scale)
        self.in_shift = torch.from_numpy(np.float32(in_shift)) if in_shift is not None else torch.zeros(self.obs_dim)
        self.in_scale = torch.from_numpy(np.float32(in_scale)) if in_scale is not None else torch.ones(self.obs_dim)
        self.out_shift = torch.from_numpy(np.float32(out_shift)) if out_shift is not None else torch.zeros(self.act_dim)
        self.out_scale = torch.from_numpy(np.float32(out_scale)) if out_scale is not None else torch.ones(self.act_dim)

    def forward(self, x, hidden_state=None):     
        self.activations = []
        
        if self.hidden_state == None:
            self.hidden_state = self.initialize_hidden_state(self.batch_size, x.device)
        
        if x.size(0) != 1 or x.size(0) != self.batch_size:
            self.hidden_state = self.initialize_hidden_state(x.size(0), x.device)

        self.batch_size = x.size(0)

        # Normalize inputs
        x = (x - self.in_shift) / (self.in_scale + 1e-8)

        # Ensure x has a batch and sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add a sequence dimension
                   
        # Pass through RNN layer
        rnn_out, self.hidden_state= self.rnn(x, self.hidden_state)

        rnn_out = self.nonlinearity(rnn_out)
        
        self.activations.append(rnn_out)
        
        out = rnn_out[:, -1, :]  # Use the output from the last time step
        
        # Pass through MLP layer
        out = self.nonlinearity(self.fc(out))
        self.activations.append(out)
        out = self.output_layer(out)

        # Scale and shift output
        out = out * self.out_scale + self.out_shift
        
        return out, hidden_state

    def close_writer(self):
        self.writer.close()

    def to(self, device):
        self.device = device
        self.in_shift = self.in_shift.to(device)
        self.in_scale = self.in_scale.to(device)
        self.out_shift = self.out_shift.to(device)
        self.out_scale = self.out_scale.to(device)
        super().to(device)
        
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the SummaryWriter from the state
        if 'writer' in state:
            del state['writer']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Recreate the SummaryWriter (won't be in pickled state)
        self.writer = SummaryWriter()
        

class GRU(nn.Module):
    def __init__(self, obs_dim, act_dim,
                 hidden_sizes=(256,256),
                lstm_hidden_size=256,  # Number of hidden units in the LSTM
                 mlp_hidden_size=256, 
                 nonlinearity='tanh',  # Activation after RNN
                 in_shift=None,
                 in_scale=None,
                 out_shift=None,
                 out_scale=None,
                 log_dir='runs/rnn_activations'):
        super(GRU, self).__init__()
        self.i = 0
        self.batch_size = 1
        self.hidden_state = None
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        assert type(hidden_sizes) == tuple
        self.layer_sizes = (obs_dim,) + hidden_sizes + (act_dim,)
        self.set_transformations(in_shift, in_scale, out_shift, out_scale)
        self.lstm = nn.GRU(input_size=obs_dim, hidden_size=lstm_hidden_size, batch_first=True)

        self.fc = nn.Linear(lstm_hidden_size, mlp_hidden_size)
        self.output_layer = nn.Linear(mlp_hidden_size, act_dim)
        
        if nonlinearity == 'relu':
            self.nonlinearity = torch.relu 
        elif nonlinearity == 'tanh':
            self.nonlinearity = torch.tanh
        elif nonlinearity == 'sigmoid':
            self.nonlinearity = torch.sigmoid
        elif nonlinearity == 'softplus':
            self.nonlinearity = nn.functional.softplus
        else:
            raise ValueError("Nonlinearity must be 'tanh', 'relu', or 'sigmoid'")

        self.activations = []
        self.writer = SummaryWriter(log_dir)
        
        
    def initialize_hidden_state(self, batch_size, device):
        return (torch.zeros(1, batch_size, 256).to(device))

    def set_transformations(self, in_shift=None, in_scale=None, out_shift=None, out_scale=None):
        self.transformations = dict(in_shift=in_shift,
                                    in_scale=in_scale,
                                    out_shift=out_shift,
                                    out_scale=out_scale)
        self.in_shift = torch.from_numpy(np.float32(in_shift)) if in_shift is not None else torch.zeros(self.obs_dim)
        self.in_scale = torch.from_numpy(np.float32(in_scale)) if in_scale is not None else torch.ones(self.obs_dim)
        self.out_shift = torch.from_numpy(np.float32(out_shift)) if out_shift is not None else torch.zeros(self.act_dim)
        self.out_scale = torch.from_numpy(np.float32(out_scale)) if out_scale is not None else torch.ones(self.act_dim)

    def forward(self, x, hidden_state=None):     
        self.activations = []
        
        if self.hidden_state == None:
            self.hidden_state = self.initialize_hidden_state(self.batch_size, x.device)
        
        if x.size(0) != 1 or x.size(0) != self.batch_size:
            self.hidden_state = self.initialize_hidden_state(x.size(0), x.device)

        self.batch_size = x.size(0)

        # Normalize inputs
        x = (x - self.in_shift) / (self.in_scale + 1e-8)

        # Ensure x has a batch and sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add a sequence dimension
                   
        # Pass through LSTM layer
        lstm_out, self.hidden_state= self.lstm(x, self.hidden_state)

        lstm_out = self.nonlinearity(lstm_out)
        
        self.activations.append(lstm_out)
        
        out = lstm_out[:, -1, :]  # Use the output from the last time step
        
        # Pass through MLP layer
        out = self.nonlinearity(self.fc(out))
        self.activations.append(out)
        out = self.output_layer(out)

        # Scale and shift output
        out = out * self.out_scale + self.out_shift
        
        return out, hidden_state

    def close_writer(self):
        self.writer.close()

    def to(self, device):
        self.device = device
        self.in_shift = self.in_shift.to(device)
        self.in_scale = self.in_scale.to(device)
        self.out_shift = self.out_shift.to(device)
        self.out_scale = self.out_scale.to(device)
        super().to(device)
        
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the SummaryWriter from the state
        if 'writer' in state:
            del state['writer']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Recreate the SummaryWriter (won't be in pickled state)
        self.writer = SummaryWriter()        

class FCNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim,
                 hidden_sizes=(64, 64),
                 nonlinearity='tanh',  # either 'tanh' or 'relu' or 'sigmoid'
                 in_shift=None,
                 in_scale=None,
                 out_shift=None,
                 out_scale=None,
                 log_dir='runs/activations'):
        super(FCNetwork, self).__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        assert type(hidden_sizes) == tuple
        self.layer_sizes = (obs_dim,) + hidden_sizes + (act_dim,)
        self.set_transformations(in_shift, in_scale, out_shift, out_scale)

        self.fc_layers = nn.ModuleList([nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1])
                                        for i in range(len(self.layer_sizes) - 1)])
        if nonlinearity == 'relu':
            self.nonlinearity = torch.relu 
        elif nonlinearity == 'tanh':
            self.nonlinearity = torch.tanh
        elif nonlinearity == 'sigmoid':
            self.nonlinearity = torch.sigmoid
        elif nonlinearity == 'softplus':
            self.nonlinearity = nn.functional.softplus
        else:
            raise ValueError("Nonlinearity must be 'tanh', 'relu', or 'sigmoid'")

        self.activations = {}
        self.writer = SummaryWriter(log_dir)

    def set_transformations(self, in_shift=None, in_scale=None, out_shift=None, out_scale=None):
        self.transformations = dict(in_shift=in_shift,
                                    in_scale=in_scale,
                                    out_shift=out_shift,
                                    out_scale=out_scale)
        self.in_shift = torch.from_numpy(np.float32(in_shift)) if in_shift is not None else torch.zeros(self.obs_dim)
        self.in_scale = torch.from_numpy(np.float32(in_scale)) if in_scale is not None else torch.ones(self.obs_dim)
        self.out_shift = torch.from_numpy(np.float32(out_shift)) if out_shift is not None else torch.zeros(self.act_dim)
        self.out_scale = torch.from_numpy(np.float32(out_scale)) if out_scale is not None else torch.ones(self.act_dim)

    def forward(self, x):
        try:
            out = x.to(self.device)
        except:
            if hasattr(self, 'device') == False:
                self.device = 'cpu'
                out = x.to(self.device)
            else:
                raise TypeError
        out = (out - self.in_shift) / (self.in_scale + 1e-8)
        for i in range(len(self.fc_layers) - 1):
            out = self.fc_layers[i](out)
            out = self.nonlinearity(out)
            self.activations["layer_"+ str(i)] = out.detach().cpu().numpy() 
        out = self.fc_layers[-1](out)
        out = out * self.out_scale + self.out_shift
        self.activations["output_layer"] = out.detach().cpu().numpy() 
        return out #Outputs final layer


    def close_writer(self):
        self.writer.close()

    def to(self, device):
        self.device = device
        # change the transforms to the appropriate device
        self.in_shift = self.in_shift.to(device)
        self.in_scale = self.in_scale.to(device)
        self.out_shift = self.out_shift.to(device)
        self.out_scale = self.out_scale.to(device)
        # move all other trainable parameters to device
        super().to(device)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the SummaryWriter from the state
        if 'writer' in state:
            del state['writer']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Recreate the SummaryWriter (won't be in pickled state)
        self.writer = SummaryWriter()

class FCNetworkWithBatchNorm(nn.Module):
    def __init__(self, obs_dim, act_dim,
                 hidden_sizes=(64, 64),
                 nonlinearity='relu',  # either 'tanh' or 'relu'
                 dropout=0,  # probability to dropout activations (0 means no dropout)
                 log_dir='runs/activations_with_batchnorm',
                 *args, **kwargs):
        super(FCNetworkWithBatchNorm, self).__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        assert type(hidden_sizes) == tuple
        self.layer_sizes = (obs_dim,) + hidden_sizes + (act_dim,)
        self.device = 'cpu'

        # hidden layers
        self.fc_layers = nn.ModuleList([nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1])
                                        for i in range(len(self.layer_sizes) - 1)])
        self.nonlinearity = torch.relu if nonlinearity == 'relu' else torch.tanh
        self.input_batchnorm = nn.BatchNorm1d(num_features=obs_dim)
        self.dropout = nn.Dropout(dropout)

        self.activations = {}
        self.writer = SummaryWriter(log_dir)

    def forward(self, x):
        out = x.to(self.device)
        out = self.input_batchnorm(out)
        for i in range(len(self.fc_layers) - 1):
            out = self.fc_layers[i](out)
            out = self.dropout(out)
            out = self.nonlinearity(out)
        out = self.fc_layers[-1](out)
        print(out)
        return out

    def hook_fn(self, module, input, output, layer_name, neuron_idx=None):
        self.activations[layer_name] = output.detach().cpu().numpy()
        self.writer.add_histogram(f'Activations/{layer_name}', output)

    def hook_fn_layer(self, module, input, output):
        layer_name = module.__class__.__name__
        self.hook_fn(module, input, output, layer_name)

    def hook_fn_neuron(self, module, input, output):
        layer_name = module.__class__.__name__
        neuron_idx = module.neuron_idx
        self.hook_fn(module, input, output, layer_name, neuron_idx)

    def register_hooks(self):
        for i, layer in enumerate(self.fc_layers):
            layer_name = f'fc_layer_{i}'
            layer.__class__.__name__ = layer_name
            layer.register_forward_hook(self.hook_fn_layer)
            

    def close_writer(self):
        self.writer.close()

    def to(self, device):
        self.device = device
        super().to(device)

    def set_transformations(self, *args, **kwargs):
        pass

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the SummaryWriter from the state
        if 'writer' in state:
            del state['writer']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Recreate the SummaryWriter (won't be in pickled state)
        self.writer = SummaryWriter()
