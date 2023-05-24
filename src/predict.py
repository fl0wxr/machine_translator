import torch
import model
import predictor

def print_metrics_per_pipeline(pipeline_name, metrics):
    print('> '+pipeline_name)
    print('>> '+'Train loss: %f | Val loss: %f'%(metrics['loss'][pipeline_name]['train'][-1], metrics['loss'][pipeline_name]['val'][-1]))
    print('>> '+'Train BLEU: %f | Val BLEU: %f'%(metrics['bleu'][pipeline_name]['train'][-1], metrics['bleu'][pipeline_name]['val'][-1]))

def user_input_loop(translator):

    while True:

        try:
            x = [input('Give a text:\n')]#['Hello world!', 'What\'s up?']

            translator(x=x, dec_mode=2, dec_config={})

            translator.display_translation()
            print()
            translator.display_n_likely(n=3)
            print()
        except KeyboardInterrupt: ## Ctrl+C triggers it
            exit('\nProcess terminated.')

        exit()

def data_sample_evaluation(translator):

    data_fpath='../datasets/pp_ell.pt'
    n_train = n_val = 10

    dataset = torch.load(f=data_fpath)
    train_src_int = dataset['train'][:][0][:n_train]
    train_tgt_int = dataset['train'][:][1][:n_train]
    val_src_int = dataset['val'][:][0][:n_val]
    val_tgt_int = dataset['val'][:][1][:n_val]

    print('\nSample from the training set:', end=2*'\n')
    translator(x=train_src_int, dec_mode=2, dec_config={})
    translator.display_translation()
    # print()
    # translator.display_n_likely(n=3)
    print(end=2*'\n')
    print('Sample from the validation set:', end=2*'\n')
    translator(x=val_src_int, dec_mode=2, dec_config={})
    translator.display_translation()
    # print()
    # translator.display_n_likely(n=3)


device = torch.device('cuda')

## The file containing a model
training_path = '../training/ell/s2s_ell_ep110.pt'

training = torch.load(training_path, map_location=device)

print('Source vocabulary size: %d'%(training['src_vocab_size']))
print('Target vocabulary size: %d'%(training['tgt_vocab_size']), end=2*'\n')

print('Current epoch is %d.'%(len(training['metrics_history']['loss']['training_pipeline']['train'])-1))
print('Translator\'s evaluation:')
print_metrics_per_pipeline(pipeline_name='training_pipeline', metrics=training['metrics_history'])
print_metrics_per_pipeline(pipeline_name='id2_pipeline', metrics=training['metrics_history'])

s2s = model.seq2seq\
(
    n_src_inp=training['src_vocab_size'],
    max_steps_src=training['max_steps_src'],
    n_tgt_inp=training['tgt_vocab_size'],
    max_steps_tgt=training['max_steps_tgt'],
    bos_int=training['tgt_vocab'].get_stoi()['<bos>'],
    eos_int=training['tgt_vocab'].get_stoi()['<eos>'],
    pad_int=training['tgt_vocab'].get_stoi()['<pad>'],
    device=device
)
s2s.load_state_dict(training['model_params'])

translator = predictor.predictor\
(
    model=s2s,
    src_vocab=training['src_vocab'],
    src_vocab_size=training['src_vocab_size'],
    max_steps_src=training['max_steps_src'],
    tgt_vocab=training['tgt_vocab'],
    tgt_vocab_size=training['tgt_vocab_size'],
    max_steps_tgt=training['max_steps_tgt'],
    tgt_name=training['dataset_name'],
    device=device
)

user_input_loop(translator=translator)
# data_sample_evaluation(translator=translator)