from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import layer


class encoder(nn.Module):

    def __init__(self, n_inp, n_max_steps, n_out, device):
        '''
        Description:
            The encoder of a MT module. The final layer is a recurrent type.

        Inputs:
            <n_inp>: Type <int>. The number of input neurons per example. Coincides with the size of the source vocabulary.
            <n_max_steps>: Type <int>. The number of source's steps including the padding size.
            <n_out>: Type <int>. For a fixed example and step, this is the number of neurons in the encoder's output layer.
            <device>: Type <torch.device>.
        '''

        super().__init__()

        self.device = device

        ## Number of embedding axes.
        self.n_emb = 600

        self.n_hid1 = 600

        self.n_inp = n_inp
        self.n_max_steps = n_max_steps

        ## Output neurons scaler. May as well be named as "self.n_hid".
        self.n_out = n_out

        self.emb = layer.dense(n_inp=self.n_inp, n_out=self.n_emb, n_steps=self.n_max_steps, device=self.device)
        self.rec1 = layer.gru(n_inp=self.n_emb, n_hid=self.n_hid1, device=self.device, dropout_rate=0.0)
        self.rec2 = layer.gru(n_inp=self.n_hid1, n_hid=self.n_out, device=self.device, dropout_rate=0.0)

    def forward(self, X, train):
        '''
        Inputs:
            <X>: Type: <torch.Tensor>. Shape: (n_examples, n_steps, vocab_size). The encoder's input which is the preprocessed source data.
            <train>: Type: <bool>.

        Outputs:
            <out>: Type: <tuple[<torch.Tensor>, <torch.Tensor>]>.
                <output_of_final_layer>: Type: <torch.Tensor>. Shape: (n_examples, n_steps, n_out). The output of the final recurrent layer containing all steps.
                <layerwise_final_states>: Type: <torch.Tensor>. Shape: (n_recurrent_layers, n_examples, n_out). Along the first axis, it contains the final state of each recurrent layer in this module.
        '''

        recurrent_layers_final_states = []

        out = self.emb(inputs=X, train=train)
        out = self.rec1(inputs=out, train=train)
        recurrent_layers_final_states.append(out[1])
        out = self.rec2(inputs=out[0], train=train)
        recurrent_layers_final_states.append(out[1])

        return (out[0], recurrent_layers_final_states)

class decoder(nn.Module):

    def __init__(self, n_tgt_inp, n_context, n_max_steps, n_out, bos_int, eos_int, pad_int, device):
        '''
        Inputs:
            <n_tgt_inp>: Type <int>. The target's size per example per step.
            <n_context>: Type <int>. The context's size.
            <n_max_steps>: Type <int>. The number of target's steps including the padding size.
            <n_out>: Type <int>. Size of the output dense layer. Equal to the size of the source's vocabulary.
            <device>: Type <torch.device>.
        '''

        super().__init__()

        self.device = device

        self.n_emb = 600
        self.n_hid1 = 600
        self.n_hid2 = 600

        self.bos_int = bos_int
        self.eos_int = eos_int
        self.pad_int = pad_int
        self.n_tgt_inp = n_tgt_inp
        self.n_context = n_context
        self.n_inp = self.n_tgt_inp
        self.n_yc = self.n_context + self.n_emb
        self.n_out = n_out
        self.n_max_steps = n_max_steps

        self.tgt_ohe_modifier = lambda x: torch.stack([F.one_hot(x[seq_idx], self.n_tgt_inp) for seq_idx in range(len(x))], axis=0).type(torch.float32).to(self.device)

        self.emb = layer.dense(n_inp=self.n_inp, n_out=self.n_emb, n_steps=self.n_max_steps, device=self.device)
        self.rec1 = layer.gru(n_inp=self.n_yc, n_hid=self.n_hid1, device=self.device, dropout_rate=0.0)
        self.rec2 = layer.gru(n_inp=self.n_hid1, n_hid=self.n_hid2, device=self.device, dropout_rate=0.0)
        self.dense_out = layer.dense(n_inp=self.n_hid2, n_out=self.n_out, n_steps=self.n_max_steps, device=self.device)
        self.softmax = nn.Softmax(dim=-1)

        self.available_pipelines = \
        [
            self.train_teacher_forcing_pipeline,
            self.train_scheduled_sampling_pipeline,
            self.prediction_stochastic_pipeline,
            self.prediction_greedy_search_pipeline,
            self.prediction_beam_search_pipeline
        ]

        self.eos_ohe = self.tgt_ohe_modifier(torch.tensor(eos_int)[None, None])
        self.pad_ohe = self.tgt_ohe_modifier(torch.tensor(pad_int)[None, None])

    ## ! Architecture: Begin

    def one_step_primary_pipeline(self, inp, context, initial_states, train):
        rec1_init_state, rec2_init_state = initial_states

        out_ = inp
        out_ = self.emb(inputs=out_, train=train)
        out_ = torch.cat((out_, context), -1)
        out_ = self.rec1(inputs=out_, H_t=rec1_init_state, train=train)
        rec1_init_state = out_[1]
        out_ = self.rec2(inputs=out_[0], H_t=rec2_init_state, train=train)
        rec2_init_state = out_[1]
        out_ = self.dense_out(inputs=out_[0], train=train)
        out_ = self.softmax(out_)

        return out_, (rec1_init_state, rec2_init_state)

    def multi_step_primary_pipeline(self, Y, context, initial_states, train):
        rec1_init_state, rec2_init_state = initial_states

        out = self.emb(inputs=Y, train=train)
        out = torch.cat((out, context), -1)
        out = self.rec1(inputs=out, H_t=rec1_init_state, train=train)
        out = self.rec2(inputs=out[0], H_t=rec2_init_state, train=train)
        out = self.dense_out(inputs=out[0], train=train)
        out = self.softmax(out)

        return out

    ## ! Architecture: End

    ## ! Wraps of the primary/main decoder's pipeline: Begin

    def train_teacher_forcing_pipeline(self, Y, context, initial_states, train, dec_config):
        '''
        Trigger:
            When <dec_mode> is 0.

        Description:
            Returns a distribution function. Given a t \in \{ 1, ..., T-1 \}, it's
            P(Y_t | Y_{t-1}=y_{t-1}).
        '''

        out = self.multi_step_primary_pipeline(Y=Y, context=context, initial_states=initial_states, train=train)

        return out

    def train_scheduled_sampling_pipeline(self, Y, context, initial_states, train, dec_config):
        '''
        Trigger:
            When <dec_mode> is 1.

        Description:
            Returns a distribution function. Given a t \in \{ 1, ..., T-1 \}, it's
            P(Y_t | Y_{t-1}=y_{t-1}).
        '''

        decay = dec_config['decay']

        switches = torch.ones(self.n_max_steps).type(torch.bool)
        switches[0:round((1-decay) * (self.n_max_steps))] = False
        switches = [True] + switches[torch.randperm(self.n_max_steps)].tolist()

        context = context[:,0:1,:]

        out = []
        for t in range(self.n_max_steps):
            if switches[t]:
                out_ = Y[:,t:t+1,:]  ## Remember that the first value is <bos> !!!

            out_, initial_states = self.one_step_primary_pipeline(inp=out_, context=context, initial_states=initial_states, train=False)
            out.append(out_[:,0,:]) ## To justify the slice out_[:,0,:]: <out_> has (n_examples, 1, n_tgt_inp), and we're iterating with respect to the step t. When the loops terminates, <out> will properly contain the time step axis.
        out = torch.stack(out).swapaxes(0, 1)

        return out

    def prediction_stochastic_pipeline(self, Y, context, initial_states, train, dec_config):
        '''
        Trigger:
            When <dec_mode> is 2.

        Description:
            Returns a distribution function.
            P(Y_t | Y_{t-1}, ..., Y_0, CONTEXT=context)
        '''

        out = []
        out_ = Y
        for t in range(self.n_max_steps):
            out_, initial_states = self.one_step_primary_pipeline(inp=out_, context=context, initial_states=initial_states, train=False)
            out.append(out_[:,0,:]) ## To justify the slice out_[:,0,:]: <out_> has (n_examples, 1, n_tgt_inp), and we're iterating with respect to the step t. When the loops terminates, <out> will properly contain the time step axis.
        out = torch.stack(out).swapaxes(0, 1)

        return out

    def prediction_greedy_search_pipeline(self, Y, context, initial_states, train, dec_config):
        '''
        Trigger:
            When <dec_mode> is 3.

        Description:
            Returns a fixed token
            x_{t+1}
            inferred from
            P(X_{t+1} | X_t=x_t, ..., X_0=x_0, CONTEXT=context)
        '''

        out = []
        out_ = Y
        for t in range(self.n_max_steps):
            out_, initial_states = self.one_step_primary_pipeline(inp=out_, context=context, initial_states=initial_states, train=False)
            out.append(out_[:,0,:])
            out_ = torch.argmax(out_, axis=-1)
            out_ = self.tgt_ohe_modifier(out_)
        out = torch.stack(out).swapaxes(0, 1)

        return out

    def prediction_beam_search_pipeline(self, Y, context, initial_states, train, dec_config):
        '''
        Trigger:
            When <dec_mode> is 4.

        Description:
            Returns a fixed token.
        '''

        def score(probability, seq_length):

            a = 0.75
            score_value = np.log(probability) / seq_length**a

            return score_value

        def get_scores(probabilities):

            scores = [[None for branch_idx in range(len(probabilities[t]))] for t in range(len(probabilities))]

            for t in range(len(probabilities)):
                seq_length = t + 1
                for branch_idx in range(len(probabilities[t])):
                    scores[t][branch_idx] = score(probabilities[t][branch_idx], seq_length)

            return scores

        def get_optimal_score_position(score):

            n_examples = len(score[0][0])

            max_score = [-np.infty for i in range(n_examples)]
            t_optimal = [None for i in range(n_examples)]
            branch_idcs_optimal = [None for i in range(n_examples)]

            for i in range(n_examples):

                for t in range(len(score)):
                    for branch_idx in range(len(score[t])):
                        if score[t][branch_idx][i] > max_score[i]:
                            max_score[i] = score[t][branch_idx][i]
                            t_optimal[i] = t
                            branch_idcs_optimal[i] = branch_idx

            return t_optimal, branch_idcs_optimal

        def get_optimal_col_sequences(beam_graph, t_optimal, branch_idcs_optimal, eos_ohe, pad_ohe):
            '''
            Description:
                Finalizes the prediction tensor containing the optimal sequences among others based on given indices.

            Inputs:
                <beam_graph>: Type: <list[<list[torch.Tensor]>]>. The outer list has length equal to n_max_steps. At the outer list's position beam_step, the inner list has length equal to the number of branches there. The tensor inside has shape (n_examples, 1, n_out). Hence a value can be indexed like this
                e.g. beam_graph[beam_step][branch_idx][example_idx, 0, out_dim]
                <t_optimal>: Type: <list[<int>]>. Length: n_examples. For a given example, this list contains the steps where beam_graph contains the optimal sequence.
                <branch_idcs_optimal>: Type: <list[<int>]>. Length: n_examples. For a given example, this list contains the branch index (relative to its corresponding step) where beam_graph contains the optimal sequence.
                <eos_ohe>: Type: <torch.Tensor>. Shape: (1, 1, n_out). The end of sequence token (<eos>) in OHE.
                <pad_ohe>: Type: <torch.Tensor>. Shape: (1, 1, n_out). The padding token (<pad>) in OHE.

            Outputs:
                <optimal_col_sequences>: Type: <torch.Tensor>. Shape: (n_examples, max_n_steps, n_out).
            '''

            n_examples = len(t_optimal)

            optimal_col_sequences = []
            for i in range(n_examples):

                ## Find optimal sequences
                optimal_sequence = beam_graph[t_optimal[i]][branch_idcs_optimal[i]][i:i+1] ## The tensor has shape (x1 example, t_optimal[i]+1, n_out).

                ## ! Fill in each sequence to achieve shape homogeneity: Begin

                empty_positions = self.n_max_steps - optimal_sequence.shape[1]

                # optimal_col_sequences.append(optimal_sequence)
                if empty_positions > 0:
                    optimal_sequence = torch.cat((optimal_sequence, eos_ohe, pad_ohe.repeat(1, empty_positions-1, 1)), axis=1)

                ## ! Fill in each sequence to achieve shape homogeneity: Begin

                optimal_col_sequences.append(optimal_sequence)

            optimal_col_sequences = torch.cat(optimal_col_sequences, axis=0)

            return optimal_col_sequences

        beam_width = dec_config['beam_width']
        assert isinstance(beam_width, int) and (beam_width >= 1) and (beam_width <= self.n_out), "E: Invalid beam width value."

        beam_graph = [[None for parent_branch_idx in range(min(beam_width, t*beam_width+1))] for t in range(self.n_max_steps+1)]
        beam_graph_probabilities = [[None for parent_branch_idx in range(min(beam_width, t*beam_width+1))] for t in range(self.n_max_steps+1)]
        beam_graph[0][0] = Y
        beam_graph_probabilities[0][0] = np.ones((Y.shape[0], 1)) ## Shape: (n_examples, 1 n_possibilities@t=0)

        for t in range(self.n_max_steps): ## Referring to the branch specified by <t> and <parent_branch_idx> as the parent branch, and all the brances connected with this parent branch that belong to step, <t>+1 as child branches.
            for parent_branch_idx in range(len(beam_graph[t])):

                ## ! Forward run: Begin

                out_ = beam_graph[t][parent_branch_idx][:,-1,:][:, None, :]
                out_, initial_states = self.one_step_primary_pipeline(inp=out_, context=context, initial_states=initial_states, train=False)

                beam_graph_node = out_

                ## ! Forward run: End

                ## ! Beam Search graph update: Begin

                if t == 0:
                    k = deepcopy(beam_width)
                else:
                    k = 1

                child_branches_token_probability_given_parent_token, child_branches_likely_tokens_int = torch.topk(input=beam_graph_node, k=k, dim=-1) ## Shapes are both (n_examples, 1 step, k)
                child_branches_token_probability_given_parent_token = child_branches_token_probability_given_parent_token[:,0,:].cpu().detach().numpy()
                child_branches_likely_tokens_ohe = self.tgt_ohe_modifier(child_branches_likely_tokens_int) ## Shape (n_examples, 1 step, k, n_out)
                parent_branch_built_sequence_probability = np.repeat(beam_graph_probabilities[t][parent_branch_idx], k, axis=1)

                child_branches_built_sequence_probability = parent_branch_built_sequence_probability * child_branches_token_probability_given_parent_token ## Probability chain rule

                child_step_connected_branch_relative_to_parent_idx = 0
                for child_step_connected_branch_idx in range(k*parent_branch_idx, k*(parent_branch_idx+1)):
                    beam_graph[t+1][child_step_connected_branch_idx] = torch.cat((beam_graph[t][parent_branch_idx], child_branches_likely_tokens_ohe[..., child_step_connected_branch_relative_to_parent_idx, :]), -2) ## Concatenating all tokens produced until this step, for this possibility. Shape (n_examples, (t+1)+1, n_out)
                    beam_graph_probabilities[t+1][child_step_connected_branch_idx] = child_branches_built_sequence_probability[..., child_step_connected_branch_relative_to_parent_idx:child_step_connected_branch_relative_to_parent_idx+1]
                    child_step_connected_branch_relative_to_parent_idx += 1
                del child_step_connected_branch_relative_to_parent_idx

                ## ! Beam Search graph update: End

        beam_graph = beam_graph[1:]

        ## ! Dropping <bos> entirely: Begin

        for t in range(self.n_max_steps):
            for parent_branch_idx in range(len(beam_graph[t])):
                beam_graph[t][parent_branch_idx] = beam_graph[t][parent_branch_idx][:,1:,:]

        beam_graph_probabilities = beam_graph_probabilities[1:]

        ## ! Dropping <bos> entirely: End

        ## ! Search optimal sequence on graph: Begin

        scores = get_scores(beam_graph_probabilities)
        t_optimal, branch_idcs_optimal = get_optimal_score_position(scores)
        optimal_col_sequences = get_optimal_col_sequences\
        (
            beam_graph=beam_graph,
            t_optimal=t_optimal,
            branch_idcs_optimal=branch_idcs_optimal,
            eos_ohe=self.eos_ohe,
            pad_ohe=self.pad_ohe
        )

        ## ! Search optimal sequence on graph: End

        return optimal_col_sequences

    ## ! Wraps of the primary/main decoder's pipeline: End

    def forward(self, Y, context, initial_states, dec_mode, dec_config, train):
        '''
        Inputs:
            <Y>: Type: <torch.Tensor> or <NoneType>. Shape: (n_examples, max_steps_tgt+1, n_tgt_inp). Targets of collection-examples in OHE starting with the <bos> token.
            <context>: Type: <torch.Tensor>. Shape: (n_examples, Y.shape[1], n_context). Context tensor. Repeated across axis 1.
            <initial_states>: Type: <torch.Tensor>. Shape: (n_recurrent_layers, n_examples, n_out). Assumed that the number of recurrent layers in the encoder is the same as in the decoder. Additionally it was assumed that the configuration of these layers are identical.
            <dec_mode>: Type: <int>. Specifies which of the decoder's pipeline will be utilized
                0: Teacher forcing pipeline
                2: Stochastic model sampling
                3: Greedy Search
                4: Beam Search
            <dec_config>: Type: <dict>. Contains the configuration values of its respective pipeline.
            <train>: Type: <bool>.

        Outputs:
            <out>: Output of corresponding pipeline.
        '''

        pipeline = self.available_pipelines[dec_mode]

        out = pipeline(Y=Y, context=context, initial_states=initial_states, dec_config=dec_config, train=train)

        return out