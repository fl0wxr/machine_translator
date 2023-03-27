import torch
from torch import nn
from torch.nn import functional as F
import codec


class seq2seq(nn.Module):
    '''
    Description:
        This is the Sequence to Sequence model based on some encoder and decoder.

    Specification (High abstraction):
        (S1) This pipeline's training process utilizes the Teacher-Forcing methodology.
        (S2) The context tensor is set to equal the final state of the encoder's output layer (which is a recurrent layer).
        (S3) In case that the model is being trained, for each given decoder's state, the input is a concatenation of the target tensor slice and the context tensor. Otherwise if the model is used to predict, for each given decoder's state, the input is a concatenation of a previous state's output with the context tensor. The decoder's initial state is the OHE of '<bos>'.
    '''

    def __init__(self, n_src_inp, max_steps_src, n_tgt_inp, max_steps_tgt, bos_int, eos_int, pad_int, device):
        '''
        Inputs:
            <n_src_inp>: Type <int>. The number of source inputs per example per step.
            <max_steps_src>: Type <int>. The number source's steps including the padding size.
            <n_tgt_inp>: Type <int>. The number of target inputs per example per step.
            <max_steps_tgt>: Type <int>. The number target's steps including the padding size.
            <bos_int>: Type <int>. The '<bos>' token expressed as an integer i.e. encoded as the target vocabularies index.
            <eos_int>: Type <int>. The '<eos>' token expressed as an integer i.e. encoded as the target vocabularies index.
            <pad_int>: Type <int>. The '<pad>' token expressed as an integer i.e. encoded as the target vocabularies index.
            <device>: Type <torch.device>.
        '''

        super().__init__()

        self.device = device

        ## Input size, coiciding with the respective vocabularies' lengths.
        self.n_src_inp = n_src_inp
        self.n_tgt_inp = n_tgt_inp

        ## Number of steps including all paddings.
        self.max_steps_src = max_steps_src
        self.max_steps_tgt = max_steps_tgt

        self.bos_int = bos_int
        self.eos_int = eos_int
        self.pad_int = pad_int

        ## These assignments are established by (S2). The "enc" in the name "enc_n_out" refers to the encoder.
        self.context_n = self.enc_n_out = 600

        self.encoder = codec.encoder(n_inp=self.n_src_inp, n_max_steps=self.max_steps_src, n_out=self.enc_n_out, device=self.device)
        self.decoder = codec.decoder(n_tgt_inp=self.n_tgt_inp, n_context=self.context_n, n_max_steps=self.max_steps_tgt, n_out=self.n_tgt_inp, device=self.device, bos_int=self.bos_int, eos_int=self.eos_int, pad_int=self.pad_int)

        self.name = 'seq2seq'

    def init_state(self, enc_output):
        '''
        Description:
            Constructs the context tensor. Assumed that the encoder returns the output of a recurrent layer along with their final states.

        Inputs:
            <enc_output>: Type: <tuple[<torch.Tensor>, <torch.Tensor>]>. The context tensor.
                <output_of_enc_final_layer>: Type: <torch.Tensor>. Shape: (n_examples, n_steps, n_out). The output of the final recurrent layer in the encoder containing all steps.
                <layerwise_final_states>: Type: <torch.Tensor>. Shape: (n_recurrent_layers, n_examples, n_out). Along the first axis, it contains the final state of each of the encoder's recurrent layers.

        Outputs:
            <context>: Type: <tuple[<torch.Tensor>, <torch.Tensor>]>. The context tensor.
                <context>: Type: <torch.Tensor>. Shape: (n_examples, n_steps, n_out). The output of the final recurrent layer in the encoder containing all steps.
                <layerwise_final_states>: Type: <torch.Tensor>. Shape: (n_recurrent_layers, n_examples, n_out). Along the first axis, it contains the final state of each of the encoder's recurrent layers.
        '''

        return enc_output

    def forward(self, dec_mode, dec_config, train, X, Y=None):
        '''
        Inputs:
            <dec_mode>: Type: <int>. Specifies which of the decoder's pipeline will be utilized.
            <dec_config>: Type: <dict>. Decoder's pipeline configuration.
            <train>: Type: <bool>.
            <X>: Type: <torch.Tensor>. Shape: (n_examples, max_steps_src, n_src_inp). Sources of collection-examples in OHE.
            <Y>: Type: <torch.Tensor> or <NoneType>. Shape: (n_examples, max_steps_tgt+1, n_tgt_inp). Targets of collection-examples in OHE starting with the <bos> token.

        Outputs:
            <Y_hat>: Type: <torch.Tensor>. Shape (n_examples, max_steps_tgt, n_tgt_inp). Predicted sequence. Estimation of Y.
        '''

        enc_out = self.encoder(X=X, train=train)
        _, initial_states = self.init_state(enc_out)
        context = initial_states[-1]

        if Y == None:
            bos_ohe = torch.zeros((X.shape[0], 1, self.n_tgt_inp), device=self.device)
            bos_ohe[:,:,self.bos_int] = 1
            Y = bos_ohe

        context = context.repeat(Y.shape[1], 1, 1).swapaxes(0, 1)

        Y_hat = self.decoder(Y=Y, context=context, initial_states=initial_states, dec_mode=dec_mode, dec_config=dec_config, train=train)

        return Y_hat