import functools
import torch
from torch.nn import functional as F
import dataset
from copy import deepcopy


def trivialize_irrelevant_possibilities(p_distr, irrelevant_possibility_indices):
    '''
    Description:
        The values corresponding to indices of <p_distr> that match the index-values specified by <irrelevant_possibility_indices> are substituted by 0. The removed probability quantity is added up and distributed equally to the rest of the probabilities. This is identical to the action of shrinking the sample space i.e. finding a *conditional* random variable.

    Inputs:
        <p_distr>: Type: <torch.Tensor>. Shape: (..., n_possibilities).
        <irrelevant_possibility_indices>: Type: <list[<int>]>.

    Outputs:
        <updated_p_distr>: Type: <torch.Tensor>. Shape: (..., n_possibilities).
    '''

    updated_p_distr = deepcopy(p_distr)

    n_possibilities = updated_p_distr.shape[-1]
    n_relevant_possibilities = n_possibilities - len(irrelevant_possibility_indices)

    redistributable_probability = torch.zeros((p_distr.shape[0], p_distr.shape[1]), device=p_distr.device)
    for irrelevant_possibility in irrelevant_possibility_indices:
        redistributable_probability += p_distr[..., irrelevant_possibility]
        updated_p_distr[..., irrelevant_possibility] = 0

    redistributable_probability /= n_relevant_possibilities

    for possibility in range(n_possibilities):
        if possibility not in irrelevant_possibility_indices:
            relevant_possibility = possibility
            updated_p_distr[..., relevant_possibility] += redistributable_probability

    return updated_p_distr

class predictor:

    def __init__(self, model, src_vocab, src_vocab_size, max_steps_src, tgt_vocab, tgt_vocab_size, max_steps_tgt, tgt_name, device):

        self.device = device

        self.model = model

        self.beam_width = 2

        self.src_vocab = src_vocab
        self.src_vocab_size = src_vocab_size
        self.max_steps_src = max_steps_src

        self.tgt_vocab = tgt_vocab
        self.tgt_vocab_size = tgt_vocab_size
        self.max_steps_tgt = max_steps_tgt

        self.tgt_name = tgt_name

        ## Source text preprocess functions
        self.initial_text_preprocess = dataset.initial_text_preprocess
        self.tokenizer = lambda x: [[t for t in f'{x} <eos>'.split(' ') if t]]
        self.src_numericalize = functools.partial(dataset.c_numericalize_str, vocab_itos=self.src_vocab.get_itos())
        self.src_c_pad_or_trim = functools.partial(dataset.c_pad_or_trim, eos_int=self.src_vocab.get_stoi()['<eos>'], pad_int=self.src_vocab.get_stoi()['<pad>'], t_bound=self.max_steps_src)
        self.src_ohe_modifier = lambda x: torch.stack([F.one_hot(x[seq_idx], self.src_vocab_size) for seq_idx in range(len(x))], axis=0).type(torch.float32).to(self.device)

    def __call__(self, x, dec_mode, dec_config):
        '''
        Inputs:
            <x>: Contains the input's information. Can be a list of strings, or a torch,tensor where its contents' shared data type may be integer or float.
            <dec_mode>: Type: <int>. Decoder's pipeline ID.
            <dec_config>: Type: <dict>. Decoder's pipeline configuration.
        '''

        self.dec_mode = dec_mode

        if isinstance(x[0], str):
            self.src_str = x
            self.src_ohe = self.preprocess(self.src_str).to(self.device)
        elif isinstance(x[0,0].item(), (int, float)):
            self.src_str = dataset.int2str(col_int=x, vocab=self.src_vocab, max_steps=self.max_steps_src)
            self.src_ohe = self.src_ohe_modifier(x)

        with torch.no_grad():
            self.p_distr = self.model(X=self.src_ohe, dec_mode=self.dec_mode, dec_config=dec_config, train=False)

        self.n_examples = len(self.p_distr)

        tgt_remove_tokens_str = ['<unk>', '<pad>', '<bos>']
        tgt_remove_tokens_int = [self.tgt_vocab.get_stoi()[token] for token in tgt_remove_tokens_str]
        self.adjusted_p_distr = trivialize_irrelevant_possibilities(self.p_distr, tgt_remove_tokens_int)

        ## Deterministic translation
        self.translated_int = dataset.ohe2int(self.adjusted_p_distr)
        self.translated_str = dataset.int2str(col_int=self.translated_int, vocab=self.tgt_vocab, max_steps=self.max_steps_tgt)

    def preprocess(self, x):

        preprocessed_x = [None for i in range(len(x))]
        for i in range(len(x)):
            preprocessed_x[i] = self.initial_text_preprocess(x[i])
            preprocessed_x[i] = self.tokenizer(preprocessed_x[i])
            preprocessed_x[i] = self.src_numericalize(preprocessed_x[i])
            preprocessed_x[i] = self.src_c_pad_or_trim(preprocessed_x[i])
            preprocessed_x[i] = torch.tensor(preprocessed_x[i])
            preprocessed_x[i] = self.src_ohe_modifier(preprocessed_x[i])[0]
        preprocessed_x = torch.stack(preprocessed_x)

        return preprocessed_x

    def display_translation(self):

        if self.n_examples == 1:
            print('Source sequence [eng]:\n%s'%(self.src_str[0]))
            print('Target sequence [%s]:\n%s'%(self.tgt_name, self.translated_str[0]))
        else:
            print(' '*4+'%-40s   %s'%('[eng]', '['+self.tgt_name+']'))
            for example_idx in range(self.n_examples):
                print('%-4s'%(str(example_idx)), end='')
                print('%-40s   '%(self.src_str[example_idx]), end='')
                print('%s'%(self.translated_str[example_idx]))

    def display_n_likely(self, n=None):
        '''
        Description:
            The probability value displayed for a given example i and step t, for the corresponding token x_{i,t} is actually
            P(x_{i,t} | x_{i,t-1}, ..., x_{i,0}).
        '''

        if (self.dec_mode == 4) and ((n == None) or (n > self.beam_width)):
            n = self.beam_width

        predicted_words_distr, predicted_words_int = torch.topk(input=self.p_distr, k=n, dim=-1) # val, ind lists of shape (self.max_steps_tgt, n) each
        predicted_words_str = \
        [
            [
                [
                    self.tgt_vocab.get_itos()[predicted_words_int[example_idx, step, possibility]]
                    for possibility in range(n)
                ]
                for step in range(self.max_steps_tgt)
            ]
            for example_idx in range(predicted_words_int.shape[0])
        ]
        predicted_words_distr = predicted_words_distr.tolist()

        for example_idx in range(len(predicted_words_str)):
            print('Source [eng]:')
            print(self.src_str[example_idx])
            print('Target token possibilities [%s]:'%(self.tgt_name))
            for step in range(self.max_steps_tgt):
                print('tkn%d: '%(step), end='')
                for possibility in range(n):
                    print('%15s: %.4f'%(predicted_words_str[example_idx][step][possibility], predicted_words_distr[example_idx][step][possibility]), end='')
                    if possibility != n-1:
                        print(' |', end='')
                if (example_idx != len(predicted_words_str)-1) or (step != self.max_steps_tgt-1):
                    print()
            if example_idx != len(predicted_words_str)-1:
                print()
        print()