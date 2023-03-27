"""
Description:
    Text parsing and processing of translations. The translations are sourced from https://www.manythings.org/anki/.

    The following special string-tokens are defined as
    '<unk>': Unknown token,
    '<pad>': Padding token,
    '<eos>': End of sequence token,
    '<bos>': Beginning of sequence token.
"""


from collections import Counter, OrderedDict
import torchtext
import ctypes
import torch
import plot


## Load the shared library
global lib
lib = ctypes.CDLL('../lib/preprocessing_tools.so')


def int2str(col_int, vocab, max_steps, trim=True, join=True):
    '''
    Description:
        Converts tokens from integer to string format.
    '''

    ## trim triggers the sequence's trimming from <eos> until the maximum step
    if trim:
        trim_condition = lambda str_seq: str_seq == '<eos>'
    else:
        trim_condition = lambda *args: False

    n_examples = col_int.shape[0]

    col_str_ = [[vocab.get_itos()[col_int[example_idx][t]] for t in range(max_steps)] for example_idx in range(n_examples)]
    col_str__ = [[] for example_idx in range(n_examples)]
    for example_idx in range(n_examples):
        for step in range(max_steps):
            if trim_condition(col_str_[example_idx][step]):
                break
            col_str__[example_idx].append(col_str_[example_idx][step])

    ## join triggers the resulting sequence's concatenation of string-tokens to form the final sentence
    if join:
        col_str = [' '.join(col_str__[example_idx]) for example_idx in range(n_examples)]
    else:
        col_str = col_str__

    return col_str

def ohe2int(p_distr):
    '''
    Description:
        Converts a given distribution to the most likely vocabulary index.
    '''

    return torch.argmax(p_distr, axis=-1)

def ohe2str(p_distr, vocab, max_steps, trim=True, join=True):

    return int2str(ohe2int(p_distr), vocab, max_steps, trim, join)

def single_gram_to_k_grams(col_int, k):
    '''
    Description:
        Converts word-tokens to multi-word, k-grams.

    Inputs:
        <col_int>: Type <list[<list[<int>]>]>. The outmost list's length equals n_examples. The innermost list's length equals the corresponding example's sequence length.
        <n>: Type <int>.

    Outputs:
        <col_multitoken_int>: Type <list[<list[<tuple[<int>]>]>]>. The first (in depth) list's length equals n_examples. The second list's length equals the corresponding example's sequence length. The tuple's length equals k.
    '''

    n_examples = len(col_int)

    if k == 1:
        col_ngram_int = col_int
        return col_ngram_int

    col_ngram_int = [[] for i in range(n_examples)]

    for i in range(n_examples):
    
        n_tokens_col_int = len(col_int[i])

        n_tokens_col_ngram_int = max(1, n_tokens_col_int - k + 1)
        for t_ in range(n_tokens_col_ngram_int):
            col_ngram_int[i].append(tuple(col_int[i][t_: t_+k]))

    return col_ngram_int

def trim_sequence(seq, eos):
    '''
    Description:
        Trims sequences the part that begins with the end of sequence token. Type <d0> is defined to either be <str> or <int>.

    Inputs:
        <seq>: Type <list[list[<d0>]]>. Length: n_examples.
        <eos>: Type <d0>. The end of sequence token.

    Outputs:
        <updated_seq>: Type <list[list[<d0>]]>. Length: n_examples.
    '''

    n_examples = len(seq)
    for i in range(n_examples):
        if eos in seq[i]:
            separator = seq[i].index(eos)
            seq[i] = seq[i][:separator]

    return seq

def purger(raw_text):
        """
        Description:
            Each line is partitioned with respect to '\t' the final member of the resulting list is dropped. This successfully gets rid of the useless last segment.

        Input:
            <raw_text>: Type: <str>.

        Output:
            <reduced_text>: Type: <str>.
        """

        reduced_text = []
        raw_sentence_seq = raw_text.split('\n')
        for instance in raw_sentence_seq:
            reduced_text.append('\t'.join(instance.split('\t')[:-1]))

        reduced_text = '\n'.join(reduced_text)

        return reduced_text

def initial_text_preprocess(text):

    # Replace non-breaking space with space
    text = text.replace('\u202f', ' ').replace('\xa0', ' ')

    # Insert space between words and punctuation marks
    no_space = lambda char, prev_char: char in ',.!?;' and prev_char != ' '
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
    for i, char in enumerate(text.lower())]

    return ''.join(out)

def tokenize(text, max_examples=None):

    src_str, tgt_str = [], []
    for i, line in enumerate(text.split('\n')):
        if max_examples and i > max_examples: break
        parts = line.split('\t')
        if len(parts) == 2:
            # Skip empty tokens
            src_str.append([t for t in f'{parts[0]} <eos>'.split(' ') if t])
            tgt_str.append([t for t in f'{parts[1]} <eos>'.split(' ') if t])

    return src_str, tgt_str

def c_numericalize_str(col_seq, vocab_itos): ## str2int
    """
    Description:
        Converts a list of string-token sequences to a list of vocabulary-index-token sequences, while maintaining their order. In other words it replicates the list, whilst replacing the string tokens with index tokens with respect to a vocabulary. It's worth noting that each sequence has a length that potentially varies and is not necessarily fixed.

    Inputs:
        <col_seq>: Type: <list[<list[<str>]>]>. The collection of sequences of words to be transformed/numericalized.
            Warning: It was assumed that for each list-member of <col_seq>, its final string-token-element has to be exactly b'<eos>', otherwise the .so file won't be able to process the input.
        <vocab_itos>: Type: <list[<str>]>. The list containing all the vocabularies string-tokens at the position with index equal to vocabulary-index-token. Must include the <unk> character in the beginning.

    Returns:
        <enc_col>: Type: <list[<list[<int>]>]>. Contains all the members of <col_seq> ("col" is for "collection") where each string replaced with the vocabularies index.
    """

    ## Prototype
    lib.arstr2num.argtypes = \
    [
        ctypes.POINTER(ctypes.POINTER(ctypes.c_char_p)), # char ***col_seq
        ctypes.c_int, # int col_seq_length
        ctypes.POINTER(ctypes.c_char_p), # char **vocab_itos
        ctypes.c_int, # vocab_itos_length
        ctypes.POINTER(ctypes.POINTER(ctypes.c_int)) # int **enc_col_seq
    ]

    ## ! Define the arrays' instances and allocate memory for them: Begin

    enc_col_seq = [[None for j in range(len(col_seq[i]))] for i in range(len(col_seq))]

    col_seq_array = (ctypes.POINTER(ctypes.c_char_p) * len(col_seq))()
    for i, seq in enumerate(col_seq):
        col_seq_array[i] = (ctypes.c_char_p * len(seq))()

    enc_col_seq_array = (ctypes.POINTER(ctypes.c_int) * len(enc_col_seq))()
    for i, enc_seq in enumerate(enc_col_seq):
        enc_col_seq_array[i] = (ctypes.c_int * len(enc_seq))()

    vocab_itos_array = (ctypes.c_char_p * len(vocab_itos))()

    ## Allocating memory
    for i, seq in enumerate(col_seq):
        for j, token_str in enumerate(seq):
            col_seq_array[i][j] = token_str.encode()

    ## Allocating memory
    for i, token_str in enumerate(vocab_itos):
        vocab_itos_array[i] = token_str.encode()

    ## ! Define the arrays' instances and allocate memory for them: End

    ## Update <enc_col_seq_array>
    lib.arstr2num(col_seq_array, len(col_seq), vocab_itos_array, len(vocab_itos), enc_col_seq_array)

    ## Copy values from <enc_col_seq_array> to <enc_col_seq>
    for i in range(len(enc_col_seq)):
        for j in range(len(enc_col_seq[i])):
            enc_col_seq[i][j] = enc_col_seq_array[i][j]

    return enc_col_seq

str2int = c_numericalize_str

def c_pad_or_trim(enc_col, eos_int, pad_int, t_bound):
    """
    Description:
        For each sequence in <enc_col>:
        1. If the sequence contains more than <t_bound> tokens, this function removes all the right side tokens with index greater than <t_bound>.
        2. If the sequence contains less than <t_bound> tokens, this function adds the integer-encoded padding token to its right side until the sequence length equals <t_bound>.

    Inputs:
        <enc_col>: Type <list[list[<int>]]>. An integer-encoded collection of sequences.
            Warning: It was assumed that for each list-member of <enc_col>, its final integer-token-element has to be exactly the integer-encoded '<eos>' given by <eos_int>, otherwise the .so file won't be able to process the input.
        <eos_int>: Type <int>. The special end-of-sequence token encoded as an integer.
        <pad_int>: Type <int>. The special padding token encoded as an integer.
        <t_bound>: Type <int>. Number of time steps for the resulting sequences.

    Outputs:
        <enc_col_>: Type <list[list[<int>]]>. Collection <enc_col>'s sequences where each sequence is padded or trimmed. All sequences have length equal to <t_bound>.
    """

    ## Prototype
    lib.pad_or_trim.argtypes = \
    [
        ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_int))
    ]

    ## ! Define the arrays' instances and allocate memory for them: Begin

    enc_col_ = [[None for word_idx in range(t_bound)] for seq_idx in range(len(enc_col))]

    enc_col_array = (ctypes.POINTER(ctypes.c_int) * len(enc_col))()
    for i, enc_seq in enumerate(enc_col):
        enc_col_array[i] = (ctypes.c_int * len(enc_seq))()

    enc_col_array_ = (ctypes.POINTER(ctypes.c_int) * len(enc_col_))()
    for i in range(len(enc_col_)):
        enc_col_array_[i] = (ctypes.c_int * t_bound)()

    ## Allocating memory
    for i in range(len(enc_col)):
        for j in range(len(enc_col[i])):
            enc_col_array[i][j] = enc_col[i][j]

    ## ! Define the arrays' instances and allocate memory for them: End

    ## Update <enc_col_array_>
    lib.pad_or_trim(enc_col_array, len(enc_col), eos_int, pad_int, t_bound, enc_col_array_)

    ## Copy values from <enc_col_seq_array> to <enc_col_seq>
    for i in range(len(enc_col_)):
        for j in range(len(enc_col_[i])):
            enc_col_[i][j] = enc_col_array_[i][j]

    return enc_col_

def build_vocab(seq_string, special_tokens):
    """
    Description:
        Generates a vocabulary object with respect to the decreasing order of token-frequency.

    Inputs:
        <seq_string>: Type: <list[<str>]>. Contains the dataset's tokens.
        <special_tokens>: Type: <list[<str>]>. Contains the special tokens.

    Outputs:
        <vocab_>: Type: <torchtext.vocab.Vocab>.
        <freqs>: Type: <list>.
    """

    seq_string_counter = Counter(seq_string)
    seq_string_sorted_by_freq_tuples = sorted(seq_string_counter.items(), key=lambda x: x[1], reverse=True)
    seq_string_ordered_dict = OrderedDict(seq_string_sorted_by_freq_tuples)
    vocab_ = torchtext.vocab.vocab\
    (
        ordered_dict = seq_string_ordered_dict,
        min_freq = 2,
        specials = special_tokens,
        special_first = True
    )

    freqs = list(iter(seq_string_ordered_dict.values()))

    return freqs, vocab_

def bosify_seq(col_seq_int, bos_int):
    '''
    Description:
        Extremely important function that takes a collection of sequences, and on the step axis, it removes the final (special) token and adds the <bos> token in the beginning of the sequence.

    Inputs:
        <col_seq_int>: Type: torch.Tensor. Shape: (n_examples, max_steps_seq).

    Outputs:
        <col_seq_int_bos>: Type: torch.Tensor. Shape: (n_examples, max_steps_seq)
    '''

    tgt_int_bos = col_seq_int[:,:-1]
    bos_int_expanded = torch.ones((col_seq_int.shape[0], 1), dtype=int) * bos_int
    col_seq_int_bos = torch.concat((bos_int_expanded, tgt_int_bos), axis=1)

    return col_seq_int_bos

class translated_text_dataset:

    def __init__(self, shuffle_seed):
        '''
        Input:
            <shuffle_seed>: Type <int>. Used as a seed value for the RNG of the associated dataset's splitting. <split_seed> must never change between training interruptions or parameter transfers. The initial seed value should be randomized.
        '''

        def parse_local_raw_data(raw_dataset_fpath):

            with open(raw_dataset_fpath, 'r') as file:
                raw_dataset_str = file.read()[:-1]

            return raw_dataset_str

        self.shuffle_seed = shuffle_seed

        self.instance_shuffle_generator = torch.Generator()
        self.instance_shuffle_generator.manual_seed(self.shuffle_seed)

        raw_dataset_fpath = '../datasets/ell.txt'
        dataset_dir_fpath = '/'.join(raw_dataset_fpath.split('/')[:-1])
        self.dataset_name = raw_dataset_fpath.split('.')[-2].split('/')[-1]

        self.max_steps_src = self.max_steps_tgt = 9
        self.n_of_instances_to_keep = -1 # For debugging use: 640

        ## [training fraction, validation fraction, test fraction].
        self.split_fractions = [0.9, 0.1, 0.0]

        self.raw_dataset_str = parse_local_raw_data(raw_dataset_fpath)

        preprocessed_dataset_fname = 'pp_' + self.dataset_name
        self.preprocessed_dataset_fpath = dataset_dir_fpath + '/' + preprocessed_dataset_fname + '.pt'

    def generate_dataset(self):
        """
        Description:
            Loads raw dataset, conducts basic token preprocessing and generates the feature and target tensors.
        """

        ## Data cleaning
        self.dataset_str = purger(self.raw_dataset_str)
        self.dataset_str = initial_text_preprocess(self.dataset_str)

        ## Tokenization (string form)
        self.src_str, self.tgt_str = tokenize(self.dataset_str)

        ## Shuffle
        perm = torch.randperm(len(self.src_str), generator=self.instance_shuffle_generator).tolist()
        self.src_str = [self.src_str[example_idx] for example_idx in perm]
        self.tgt_str = [self.tgt_str[example_idx] for example_idx in perm]

        self.src_str = self.src_str[:self.n_of_instances_to_keep]
        self.tgt_str = self.tgt_str[:self.n_of_instances_to_keep]

        ## Vocabulary and token-wise frequency of appearance
        src_freqs, self.src_vocab = build_vocab\
        (
            [token for sentence in self.src_str for token in sentence],
            special_tokens=['<unk>', '<pad>', '<eos>']
        )
        tgt_freqs, self.tgt_vocab = build_vocab\
        (
            [token for sentence in self.tgt_str for token in sentence],
            special_tokens=['<unk>', '<pad>', '<eos>', '<bos>']
        )

        ## Encode tokens from strings to integers
        self.src_int = c_numericalize_str(self.src_str, self.src_vocab.get_itos()) ## = X
        self.tgt_int = c_numericalize_str(self.tgt_str, self.tgt_vocab.get_itos()) ## = Y

        ## Produce plots
        plot.plot_sentence_size(data_pair=(self.src_str, self.tgt_str), image_name=self.dataset_name+'_cnt_examples_per_sentence_len')
        plot.plot_frequency_curves(freqs_pair=(src_freqs, tgt_freqs), image_name=self.dataset_name+'_freq')

        ## Pad or trim and conversion to tensor
        self.src_int = torch.tensor\
        (
            c_pad_or_trim\
            (
                enc_col=self.src_int,
                eos_int=self.src_vocab.get_stoi()['<eos>'],
                pad_int=self.src_vocab.get_stoi()['<pad>'],
                t_bound=self.max_steps_src
            ),
            # dtype=torch.int32
        )
        self.tgt_int = torch.tensor\
        (
            c_pad_or_trim\
            (
                enc_col=self.tgt_int,
                eos_int=self.tgt_vocab.get_stoi()['<eos>'],
                pad_int=self.tgt_vocab.get_stoi()['<pad>'],
                t_bound=self.max_steps_tgt
            ),
            # dtype=torch.int32 ## This leads to an error in <torch.nn.functional.one_hot> because <self.tgt_int> stops being an "index tensor" if its data type is a 32-bit integer.
        )

        # self.eos_test()
        # self.test_correct_sentence()

        self.n_instances = self.src_int.shape[0]

        self.src_vocab_size = len( self.src_vocab.get_itos() )
        self.tgt_vocab_size = len( self.tgt_vocab.get_itos() )

        self.tgt_int_bos = bosify_seq(col_seq_int=self.tgt_int, bos_int=self.tgt_vocab.get_stoi()['<bos>'])

        self.dataset = torch.utils.data.TensorDataset(self.src_int, self.tgt_int, self.tgt_int_bos)

        separateXy = lambda set: ( torch.stack([set[i][_] for i in range(len(set))], axis=0) for _ in range(2) )

        if self.split_fractions[-1] != 0: # Included a test set
            self.train_set, self.val_set, self.test_set = torch.utils.data.random_split(self.dataset, self.split_fractions, generator=torch.Generator().manual_seed(42))

            self.n_train = len(self.train_set)
            self.n_val = len(self.val_set)
            self.n_test = len(self.test_set)

            self.X_train, self.y_train = separateXy(self.train_set)
            self.X_val, self.y_val = separateXy(self.val_set)
            self.X_test, self.y_test = separateXy(self.test_set)

            dataset = {'train': self.train_set, 'val': self.val_set, 'test': self.test_set}

        else: # Excluded the test set
            self.train_set, self.val_set = torch.utils.data.random_split(self.dataset, self.split_fractions[:2], generator=torch.Generator().manual_seed(42))

            self.n_train = len(self.train_set)
            self.n_val = len(self.val_set)

            self.X_train, self.y_train = separateXy(self.train_set)
            self.X_val, self.y_val = separateXy(self.val_set)

            dataset = {'train': self.train_set, 'val': self.val_set, 'test': None}

        torch.save(obj=dataset, f=self.preprocessed_dataset_fpath)

    def eos_test(self):
        '''
        Description:
            <eos> token exists exactly once iff it's a pass.
        '''

        pass_eos_test = True
        for i in range(self.src_int.shape[0]):
            exactly_one = (Counter(self.src_int[i,:].tolist())[self.src_vocab.get_stoi()['<eos>']] == 1) and (Counter(self.tgt_int[i,:].tolist())[self.tgt_vocab.get_stoi()['<eos>']] == 1)
            pass_eos_test = pass_eos_test and exactly_one

        if pass_eos_test:
            print('eos test passed')
        else:
            print('eos test failed')

    def test_correct_sentence(self):

        # translated_int = torch.argmax(p_distr, axis=2)[0]
        translated_str_src = [self.src_vocab.get_itos()[self.src_int[100,t]] for t in range(self.src_int.shape[1])]
        translated_str_tgt = [self.tgt_vocab.get_itos()[self.tgt_int[100,t]] for t in range(self.tgt_int.shape[1])]