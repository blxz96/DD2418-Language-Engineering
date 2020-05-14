import torch
from torch import nn
import numpy as np


class GRUCellV2(nn.Module):
    """
    GRU cell implementation
    """
    def __init__(self, input_size, hidden_size, activation=torch.tanh):
        """
        Initializes a GRU cell

        :param      input_size:      The size of the input layer
        :type       input_size:      int
        :param      hidden_size:     The size of the hidden layer
        :type       hidden_size:     int
        :param      activation:      The activation function for a new gate
        :type       activation:      callable
        """
        # input size = 10
        # hidden size = 20
        # batch = 5
        # sequence length = 3

        super(GRUCellV2, self).__init__()
        self.activation = activation
        
        # initialize weights by sampling from a uniform distribution between -K and K
        K = 1 / np.sqrt(hidden_size) #0.2236 with current seed

        # weights
        self.w_ih = nn.Parameter(torch.rand(3 * hidden_size, input_size) * 2 * K - K)  # Size = 60 x 10. Note: w_ih is made up of w_ir, w_iz, w_in. Each is 20 x 10. 
        self.w_hh = nn.Parameter(torch.rand(3 * hidden_size, hidden_size) * 2 * K - K) # Size = 60 x 20. Note: w_hh is made up of w_hr, w_hz, w_hn. Each is 20 x 10.
        self.b_ih = nn.Parameter(torch.rand(3 * hidden_size, 1) * 2 * K - K)           # Size = 60 x 1 . Note: b_ih is made up of b_ir, b_iz, b_in. Each is 20 x 1.
        self.b_hh = nn.Parameter(torch.rand(3 * hidden_size, 1) * 2 * K - K)           # Size = 60 x 1 . Note: b_hh is made up of b_hr, b_hz, b_hn. Each is 20 x 1.
        
    def forward(self, x, h):
        """
        Performs a forward pass through a GRU cell


        Returns the current hidden state h_t for every datapoint in batch.
        
        :param      x:    an element x_t in a sequence
        :type       x:    torch.Tensor
        :param      h:    previous hidden state h_{t-1}
        :type       h:    torch.Tensor
        """
        
        # YOUR CODE HERE
        
        # print(x.size()) # 5 x 10 =  batch size x input size
        # print(h.size()) # 5 x 20 =  batch size x hidden size
        batch_size = h.size()[0]

        w_ih = torch.matmul(x, self.w_ih.T)                      # Size = 5 x 60 
        w_ir, w_iz, w_in= torch.chunk(w_ih, 3, 1)                # Note: w_ih is made up of w_xr, w_xz and w_xn. Each is 5 x 20.

        w_hh = torch.matmul(h, self.w_hh.T)                      # Size = 5 x 60
        w_hr, w_hz, w_hn = torch.chunk(w_hh, 3, 1)               # Note: w_hh is made up of w_hr, w_hz and w_hn. Each is 5 x 20.

        # Size of b_ih = 60 x 1. torch.repeat_interleave can be used to made to broadcast size to 60 x 5. Then transpose to become 5 x 60.
        # Using torch.chunk, b_ir, b_iz, b_in will then have the size of 5 x 20
        b_ir, b_iz, b_in = torch.chunk(torch.repeat_interleave(self.b_ih, repeats = batch_size, dim = 1).T, 3, 1) 

        # Size of b_hh = 60 x 1. torch.repeat_interleave can be used to made to broadcast size to 60 x 5. Then transpose to become 5 x 60.
        # Using torch.chunk, b_hr, b_hz and b_hn will then have the size of 5 x 20
        b_hr, b_hz, b_hn = torch.chunk(torch.repeat_interleave(self.b_hh, repeats = batch_size, dim = 1).T, 3, 1)

        # Reset gate
        r_t = torch.sigmoid(w_hr + b_hr + w_ir + b_ir)

        # Update(input) gate
        z_t = torch.sigmoid(w_hz + b_hz + w_iz + b_iz)

        # Tentative new hidden state
        n_t = torch.tanh(r_t * (w_hn + b_hn) + w_in + b_in)

        # New hidden state
        # According to slides should be return ((1 - z_t) * h + z_t * n_t)
        # However, according to pytorch documentation, it is return n_t + z_t * (h - n_t) = ((1 - z_t) * n_t + z_t * h)
        return n_t + z_t * (h - n_t)
        


class GRU2(nn.Module):
    """
    GRU network implementation
    """
    def __init__(self, input_size, hidden_size, bias=True, activation=torch.tanh, bidirectional=False):
        super(GRU2, self).__init__()
        self.bidirectional = bidirectional
        self.fw = GRUCellV2(input_size, hidden_size, activation=activation) # forward cell
        if self.bidirectional:
            self.bw = GRUCellV2(input_size, hidden_size, activation=activation) # backward cell
        self.hidden_size = hidden_size
        
    def forward(self, x):
        """
        Performs a forward pass through the whole GRU network, consisting of a number of GRU cells.
        Takes as input a 3D tensor `x` of dimensionality (B, T, D),
        where B is the batch size;
              T is the sequence length (if sequences have different lengths, they should be padded before being inputted to forward)
              D is the dimensionality of each element in the sequence, e.g. word vector dimensionality

        The method returns a 3-tuple of (outputs, h_fw, h_bw), if self.bidirectional is True,
                           a 2-tuple of (outputs, h_fw), otherwise
        `outputs` is a tensor containing the output features h_t for each t in each sequence (the same as in PyTorch native GRU class);
                  NOTE: if bidirectional is True, then it should contain a concatenation of hidden states of forward and backward cells for each sequence element.
        `h_fw` is the last hidden state of the forward cell for each sequence, i.e. when t = length of the sequence;
        `h_bw` is the last hidden state of the backward cell for each sequence, i.e. when t = 0 (because the backward cell processes a sequence backwards)
        
        :param      x:    a batch of sequences of dimensionality (B, T, D)
        :type       x:    torch.Tensor
        """
        
        # YOUR CODE HERE
        
        torch.autograd.set_detect_anomaly(True)

        B = x.size()[0] # batch size, 5
        T = x.size()[1] # seq length, 3
        D = x.size()[2] # no. of features, 10

        outputs = torch.zeros((B, T, self.hidden_size)) # Size = 5 x 3 x 20

        # x.size() = 5 x 3 x 10
        # output.size() = 5 x 3 x 20
        
        for t in range(T):# 0,1,2
            pre = torch.zeros((B, self.hidden_size)) if t == 0 else outputs.clone()[:, t-1, :] # pre.size() = 5 x 20
            outputs[:, t, :] = self.fw.forward(x[:, t, :], pre)                        # x[:, t, :].size() = 5 x 10, output[:, t, :].size() = 5 x 20
            

        if self.bidirectional:
            outputs_rev = torch.zeros((B, T, self.hidden_size)) # Size = 5 x 3 x 20
            for t in range(T-1, -1, -1): # 2,1,0
                pre = torch.zeros((B, self.hidden_size)) if t == T-1 else outputs_rev.clone()[:, t+1, :] # Size = 5 x 20
                outputs_rev[:, t, :] = self.bw.forward(x[:, t, :], pre)                          # x[:, t, :].size() = 5 x 10, output[:, t, :].size() = 5 x 20

        if self.bidirectional:
            # returning outputs, h_fw, h_bw
            return outputs, outputs[:, -1, :], outputs_rev[:, 0, :]
        else:
            # returning outputs, h_fw
            return outputs, outputs[:, -1, :]



def is_identical(a, b):
    return "Yes" if np.all(np.abs(a - b) < 1e-6) else "No"


if __name__ == '__main__':
    torch.manual_seed(100304343)
    x = torch.randn(5, 3, 10)                                       
    gru = nn.GRU(10, 20, bidirectional=False, batch_first=True) # if batch_first = True, then the input and output tensors are provided as (batch, seq, feature)
    outputs, h = gru(x)
    #print(outputs)
    #print(h)

    torch.manual_seed(100304343)
    x = torch.randn(5, 3, 10)
    gru2 = GRU2(10, 20, bidirectional=False)
    outputs, h_fw = gru2(x)
    #print(outputs)
    #print(h_fw)
    
    print("Checking the unidirectional GRU implementation")
    print("Same hidden states of the forward cell?\t\t{}".format(
        is_identical(h[0].detach().numpy(), h_fw.detach().numpy())
    ))

    torch.manual_seed(100304343)
    x = torch.randn(5, 3, 10)
    gru = GRU2(10, 20, bidirectional=True)
    outputs, h_fw, h_bw = gru(x)
    #print(outputs)
    #print(h_fw)
    #print(h_bw)

    torch.manual_seed(100304343)
    x = torch.randn(5, 3, 10)
    gru2 = nn.GRU(10, 20, bidirectional=True, batch_first=True)
    outputs, h = gru2(x)
    
    print("Checking the bidirectional GRU implementation")
    print("Same hidden states of the forward cell?\t\t{}".format(
        is_identical(h[0].detach().numpy(), h_fw.detach().numpy())
    ))
    print("Same hidden states of the backward cell?\t{}".format(
        is_identical(h[1].detach().numpy(), h_bw.detach().numpy())
    ))