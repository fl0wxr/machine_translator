import torch
from torch import nn
import math
import weight_initializer


class dense(nn.Module):
    '''
    Description:
        Positionwise dense layer. A generalization of the traditional dense-fully-connected layer which instead assumes an input of shape (n_examples, n_inp). The current layer applies that same operation for each input step, by using the same trainable parameters on each input step.
    '''

    def __init__(self, n_inp, n_out, n_steps, device, dropout_rate=0.0):

        super().__init__()

        self.device = device

        self.n_inp = n_inp
        self.n_out = n_out
        self.n_steps = n_steps

        self.dropout_rate = dropout_rate

        self.W = weight_initializer.xavier_uniform(self.n_inp, self.n_out, self.device)
        self.b = nn.Parameter(torch.zeros(self.n_out).to(self.device))

        self.name = 'dense'

    def forward(self, inputs, train):
        '''
        Inputs:
            <inputs>: Type: <torch.Tensor>. Shape: (n_examples, n_steps, n_inp) or (n_examples, n_inp).
            <train>: Type: <bool>.

        Outputs:
            <outputs>: Type: <torch.Tensor>. Shape: (n_examples, n_steps, n_out) if len(inputs.shape) is 3 or (n_examples, n_out) if len(inputs.shape) is 3.
        '''

        input_shape_length = len(inputs.shape)
        if input_shape_length == 2:
            ## Expanding inputs to a tensor with shape (n_examples, 1, n_inp)
            inputs = inputs[:, None, :]

        outputs = []
        for (t, X_t) in enumerate(inputs.swapaxes(0,1)):
            O_t = torch.matmul(X_t, self.W) + self.b
            outputs.append(O_t)
        outputs = torch.stack(outputs).swapaxes(0,1)

        if train and (self.dropout_rate != 0.0):
            switches = torch.ones(outputs.shape[2]).to(self.device)
            switches[0:round(self.dropout_rate * outputs.shape[2])] = 0.0
            switches = switches[torch.randperm(outputs.shape[2])]

            switches_outputs = switches.repeat(outputs.shape[0], outputs.shape[1], 1)
            outputs = outputs * switches_outputs

        if input_shape_length == 2:
            outputs = outputs[:, 0, :]

        return outputs

    def __repr__(self):

        n_pars = 0
        for par in self.parameters():
            n_pars += math.prod(par.shape)

        return f"%s(n_params=%d, n_inp=%d, n_out=%d, n_steps=%d)"%(self.name, n_pars, self.n_inp, self.n_out, self.n_steps)

class vanilla_recurrent(nn.Module):
    """
    Description:
        Vanilla recurrent layer.
    """

    def __init__(self, n_inp, n_hid, device):

        super().__init__()

        self.device = device

        self.n_inp = n_inp
        self.n_hid = n_hid

        self.W_xh = weight_initializer.xavier_uniform(self.n_inp, self.n_hid, self.device)
        self.W_hh = weight_initializer.xavier_uniform(self.n_hid, self.n_hid, self.device)
        self.b_h = nn.Parameter(torch.zeros(n_hid).to(self.device))

        self.name = 'vanilla_recurrent'

    def forward(self, inputs, H_t=None):
        """
        Inputs:
            <inputs>: Type: <torch.Tensor>. Shape: (n_minibatch, n_steps, n_inp).
            <H_t>: Type: <torch.Tensor>. Shape: (n_minibatch, n_hid).

        Outputs:
            <outputs>: Type: torch.Tensor. Shape: (n_minibatch, n_steps, n_hid).
        """

        if H_t is None:
            H_t = torch.zeros((inputs.shape[0], self.n_hid)).to(self.device)
        outputs = []
        for (t, X_t) in enumerate(inputs.swapaxes(0, 1)):
            H_t = torch.tanh(torch.matmul(X_t, self.W_xh) + torch.matmul(H_t, self.W_hh) + self.b_h)
            outputs.append(H_t)
        outputs = torch.stack(outputs).swapaxes(0, 1)

        return outputs, H_t
    
    def __repr__(self):

        n_pars = 0
        for par in self.parameters():
            n_pars += math.prod(par.shape)

        return f"%s(n_params=%d)"%(self.name, n_pars)

class gru(nn.Module):

    def __init__(self, n_inp, n_hid, device, dropout_rate=0.0):

        def init_weight(n_in, n_out):

            bound = math.sqrt(3.0) * math.sqrt(2.0 / float(n_in + n_out))

            return nn.Parameter(torch.rand((n_in, n_out)).to(self.device)*(2*bound)-bound)

        super().__init__()

        self.device = device

        self.n_inp = n_inp
        self.n_hid = n_hid

        self.dropout_rate = dropout_rate

        triple = lambda: \
        (
            weight_initializer.xavier_uniform(self.n_inp, self.n_hid, self.device),
            weight_initializer.xavier_uniform(self.n_hid, self.n_hid, self.device),
            nn.Parameter(torch.zeros(self.n_hid).to(self.device))
        )
        self.W_xz, self.W_hz, self.b_z = triple() # Update gate
        self.W_xr, self.W_hr, self.b_r = triple() # Reset gate
        self.W_xh, self.W_hh, self.b_h = triple() # Candidate hidden state

        self.name = 'gru'

    def forward(self, inputs, train, H_t=None):
        '''
        Inputs:
            <inputs>: Type: <torch.Tensor>. Shape: (n_examples, n_steps, vocab_size).
            <train>: Type: <bool>. Toggles training/predicting.
            <H_t>: Type: <torch.Tensor>. Shape: (n_examples, n_hid). The initial hidden state. Default is None.

        Outputs:
            <outputs_>: Type: <tuple(<outputs>, <H_t>)>. Assuming that T=n_steps.
                <outputs>: <torch.Tensor>. Shape: (n_examples, n_steps, n_hid). Hidden states for all t=0,1,...,T.
                <H_t>: <torch.Tensor>. Shape: (n_examples, n_hid). Final hidden state at t=T.
        '''

        if H_t is None:
            # Initial state with shape: (self.batch_size, self.n_hid)
            H_t = torch.zeros((inputs.shape[0], self.n_hid), device=self.device)
        outputs = []
        for (t, X_t) in enumerate(inputs.swapaxes(0, 1)):
            Z_t = torch.sigmoid(torch.matmul(X_t, self.W_xz) + torch.matmul(H_t, self.W_hz) + self.b_z)
            R_t = torch.sigmoid(torch.matmul(X_t, self.W_xr) + torch.matmul(H_t, self.W_hr) + self.b_r)
            H_tilde_t = torch.tanh(torch.matmul(X_t, self.W_xh) + torch.matmul(R_t * H_t, self.W_hh) + self.b_h)
            H_t = Z_t * H_t + (1 - Z_t) * H_tilde_t
            outputs.append(H_t)
        outputs = torch.stack(outputs).swapaxes(0, 1)

        if train and (self.dropout_rate != 0.0):

            switches = torch.ones(self.n_hid).to(self.device)
            switches[0:round(self.dropout_rate * self.n_hid)] = 0.0
            switches = switches[torch.randperm(self.n_hid)]

            switches_outputs = switches.repeat(outputs.shape[0], outputs.shape[1], 1)
            outputs = outputs * switches_outputs

            switches_H_t = switches.repeat(H_t.shape[0], 1)
            H_t = H_t * switches_H_t

        return outputs, H_t

    def __repr__(self):

        n_pars = 0
        for par in self.parameters():
            n_pars += math.prod(par.shape)

        return f"%s(n_params=%d, n_inp=%d, n_hid=%d)"%(self.name, n_pars, self.n_inp, self.n_hid)