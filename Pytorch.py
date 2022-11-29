import numpy as np
import torch
import torchvision
# import torchvision.transforms as transforms
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class GausianNoise(torch.nn.Module):
    
    def __init__(self, sigma=0.01, is_relative_detach=True):
        super(GausianNoise, self).__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).to(device)
        
    def forward(self, x):
        if self.training and (self.sigma != 0):
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.expand(*x.size()).float().normal_() * scale
            x += sampled_noise
        return x
    
    
    
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # check whether we are in training mode.
    if not torch.is_grad_enabled:
        # in prediction mode, use mean and variance obtained by moving avg.
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2,4)
        if len(X.shape) == 2:
            mean = X.mean(dim=0)
            var  = ((X - mean) ** 2).mean(dim=0)
        else:
            # When using two dimensional convolution layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we 
            # need to maintain the shape of X, so the broadcasting 
            # operation can be carried out later
            mean = X.mean(dim=(0,2,3), keepdim=True)
            var  = ((X - mean) ** 2).mean(dim=(0,2,3), keepdim=True)
        # In training mode the current mean and variance are used.
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update the mean and variance using moving average.
        moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
        moving_var  = (1.0 - momentum) * moving_var + momentum * var
    # Scale and shift
    Y = gamma * X_hat + beta
    
    return Y, moving_mean.data, moving_var.data




class BatchNorm(torch.nn.Module):
    
    def __init__(self, num_features, num_dims=4):
        super(BatchNorm, self).__init__()
        if (num_dims==2):
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale and shift parameters (model parameters) are initialized to 1 and 0 respectively.
        self.gamma = torch.nn.Parameter(torch.ones(shape))    # scale parameters
        self.beta  = torch.nn.Parameter(torch.zeros(shape))
        # Variables which are not model parameters are initialized to 0 and 1 respectively
        self.moving_mean     = torch.zeros(shape)  # Rolling mean
        self.moving_variance = torch.ones(shape)  # Rolling Standard deviation
    
    def forward(self, x):
        if self.moving_mean.device != x.device:
            self.moving_mean = self.moving_mean.to(x.device)
            self.moving_variance = self.moving_variance.to(x.device)
        # save the updated moving mean and moving variance
        Y, self.moving_mean, self.moving_variance = batch_norm(x, self.gamma, 
                                                               self.beta, self.moving_mean, 
                                                               self.moving_variance, eps=1e-5, 
                                                               momentum=0.1)
        return Y
    
    
    
class Convolution(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=(0,0)):
        super(Convolution, self).__init__()
        self.in_channels, self.out_channels, self.kernel_size, self.kernel_size = in_channels, out_channels, kernel_size[0], kernel_size[1]
        self.stride, self.pad, self.pad = stride, padding[0], padding[1]
        self.ker = torch.nn.Parameter(torch.empty(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        torch.nn.init.kaiming_uniform_(self.ker, a=np.math.sqrt(5))
    
    
    def zero_pad(self, img, pad, mode='constant', value=0):
        X_pad = torch.nn.functional.pad(img, (pad,pad, pad, pad, 0,0, 0,0), mode=mode, value=value)
        return X_pad
    
    
    def forward(self, img):
        self.img_batch, self.num_channels, self.img_height, self.img_width = img.shape
        self.out_height = int((self.img_height - self.kernel_size + 2*self.pad) / self.stride) + 1
        self.out_width = int((self.img_width - self.kernel_size + 2*self.pad) / self.stride) + 1
        pad_img = self.zero_pad(img, self.pad)
        
        unfolded = torch.nn.functional.unfold(pad_img, kernel_size=(self.kernel_size, self.kernel_size))
        conv = unfolded.transpose(1, 2).matmul(self.ker.view(self.ker.size(0), -1).t()).transpose(1, 2)
        out = torch.nn.functional.fold(conv, (self.out_height, self.out_width), kernel_size=(1,1))
        return out
    
class Dense(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, bias=True):
        super(Dense, self).__init__()
        self.weights = torch.nn.Parameter(torch.empty((out_channels, in_channels)))
        self.biases  = torch.nn.Parameter(torch.empty(out_channels,))
        torch.nn.init.xavier_uniform_(self.weights, gain=1.)
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1./np.math.sqrt(fan_in)
        torch.nn.init.normal_(self.biases, -bound, bound)
        
    def forward(self, inputs):
        x = torch.add(torch.matmul(inputs, self.weights.T), self.biases)
        return x
    
    
class ConvNet(torch.nn.Module):
    
    def __init__(self,):
        super(ConvNet, self).__init__()
        self.conv1        = Convolution(1, 32, (3, 3), padding=(1,1))
        self.conv1_bnn    = BatchNorm(32)
        
        self.conv2        = Convolution(32, 32, (5, 5), padding=(2,2))
        self.conv2_bnn    = BatchNorm(32)
        
        self.conv3        = Convolution(32, 64, (5,5), padding=(2,2))
        self.conv3_bnn    = BatchNorm(64)
        
        self.conv4         = Convolution(64, 64, (5,5), padding=(2,2))
        self.conv4_bnn    = BatchNorm(64)
        
        self.conv5         = Convolution(64, 128, (5,5), padding=(2,2))
        self.conv5_bnn     = BatchNorm(128)
        
        self.pool         = torch.nn.MaxPool2d((2, 2),2)
        self.fc1          = Dense(128*7*7, 120)
        self.fc1_bnn      = BatchNorm(120, num_dims=2)
        self.fc2          = Dense(120, 84)
        self.fc2_bnn      = BatchNorm(84, num_dims=2)
        self.fc3          = Dense(84, 10)
        self.dropout_1    = torch.nn.Dropout(0.2)
        self.dropout_2    = torch.nn.Dropout(0.3)
        self.gaussian_noise1 = GausianNoise(0.1) 
        self.gaussian_noise2 = GausianNoise(0.01)
        
        
    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.conv1_bnn(x)
        x = self.gaussian_noise1(x)
        x = self.conv2_bnn(torch.nn.functional.relu(self.conv2(x)))
        x = self.gaussian_noise1(x)
        x = self.pool(x)
        x = self.dropout_2(x)
        x = self.conv3_bnn(torch.nn.functional.relu(self.conv3(x)))
        x = self.conv4_bnn(torch.nn.functional.relu(self.conv4(x)))
        x = self.gaussian_noise2(x)
        x = self.pool(x)
        x = self.dropout_2(x)
        x = self.conv5_bnn(torch.nn.functional.relu(self.conv5(x)))
        x = x.view(-1, 128*7*7)
        x = self.dropout_2(self.fc1_bnn(torch.nn.functional.relu(self.fc1(x))))
        x = self.gaussian_noise2(x)
        x = self.dropout_1(self.fc2_bnn(torch.nn.functional.relu(self.fc2(x))))
        x = self.gaussian_noise2(x)
        x = self.fc3(x)
        return x