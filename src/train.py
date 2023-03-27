import random
import torch
import trainer
import model
import dataset


def train_from_scratch(device):

    data = dataset.translated_text_dataset(shuffle_seed=5000) # shuffle_seed=random.randint(-10**10, 10**10)
    data.generate_dataset()

    print('Source vocabulary size: %d'%(data.src_vocab_size))
    print('Target vocabulary size: %d'%(data.tgt_vocab_size))

    s2s = model.seq2seq\
    (
        n_src_inp=data.src_vocab_size,
        max_steps_src=data.max_steps_src,
        n_tgt_inp=data.tgt_vocab_size,
        max_steps_tgt=data.max_steps_tgt,
        bos_int=data.tgt_vocab.get_stoi()['<bos>'],
        eos_int=data.tgt_vocab.get_stoi()['<eos>'],
        pad_int=data.tgt_vocab.get_stoi()['<pad>'],
        device=device
    )
    print(s2s)

    trainer_ = trainer.rnn_trainer(model_=s2s, data=data, device=device)
    trainer_.train()

def load_and_train_model(training_path, device):

    training = torch.load(training_path, map_location=device)

    data = dataset.translated_text_dataset(shuffle_seed=training['shuffle_seed'])
    data.generate_dataset()

    print('Source vocabulary size: %d'%(data.src_vocab_size))
    print('Target vocabulary size: %d'%(data.tgt_vocab_size))

    s2s = model.seq2seq\
    (
        n_src_inp=data.src_vocab_size,
        max_steps_src=data.max_steps_src,
        n_tgt_inp=data.tgt_vocab_size,
        max_steps_tgt=data.max_steps_tgt,
        bos_int=data.tgt_vocab.get_stoi()['<bos>'],
        eos_int=data.tgt_vocab.get_stoi()['<eos>'],
        pad_int=data.tgt_vocab.get_stoi()['<pad>'],
        device=device
    )
    s2s.load_state_dict(training['model_params'])
    rnn_trainer_ = trainer.rnn_trainer(model_=s2s, data=data, device=device)
    rnn_trainer_.train(metrics_history=training['metrics_history'])


device = torch.device('cuda')

train_from_scratch(device=device)
# load_and_train_model(training_path='../training/seq2seq_live_ep781.pt', device=device)