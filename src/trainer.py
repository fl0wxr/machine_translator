import os
from copy import deepcopy
from time import time
import datetime
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import metrics
import plot
import dataset
import math


class rnn_trainer(nn.Module):

    def __init__(self, model_, data, device):
        """
        Inputs:
            <model_>: Type: <torch.nn.module>. Specifies the trained model's architecture and parameters.
            <data>: Type: <class>. Contains all the necessary dataset's information.
            <device>: Type: <torch.device>.
        """

        super().__init__()

        self.device = device

        self.model = model_
        self.data = data

        self.epochs = 300
        self.lr = 0.005
        self.minibatch_size = 2**10

        # self.decay_rate = 16.666 # DEFAULT: 16.666
        # self.min_decay = 5
        # self.decay_scheduler = lambda epoch: max(self.min_decay, self.decay_rate/(self.decay_rate+math.exp(epoch/self.decay_rate)))
        self.decay_scheduler = lambda epoch: 9/9

        ## Backup frequence used for live training monitoring
        self.bkp_freq = 10

        ## Backup frequency of saved results
        saved_bkp_freq = 10
        self.scheduled_checkpoints = {}#{10, 20, 30, 60, 90, 120, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800, 1000} #{saved_bkp_freq*ep for ep in range(self.epochs // saved_bkp_freq)}; self.scheduled_checkpoints.remove(0)

        self.metric_names = ['loss', 'bleu']
        self.pipeline_names = ['training_pipeline', 'id2_pipeline']
        self.data_subset_names = ['train', 'val']

        self.training_dir_path, self.training_format = '../training/', '.pt'
        self.criterion = metrics.categ_cross_entropy ## Has to be masked
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)

        self.train_dataloader = torch.utils.data.DataLoader(self.data.train_set, batch_size=self.minibatch_size, shuffle=True)
        self.val_dataloader = torch.utils.data.DataLoader(self.data.val_set, batch_size=2**10, shuffle=False)

        ## One hot encoder of tensor <tsr> having shape (n_examples, n_steps, n) and number of classes <n_classes>.
        self.ohe_modifier = lambda tsr, n_classes: torch.stack([F.one_hot(tsr[:,seq_idx], n_classes) for seq_idx in range(tsr.shape[-1])], axis=0).swapaxes(0, 1).type(torch.float32).to(self.device)

        n_examples_display = 5
        X_train_small_idcs = torch.round(torch.linspace(51, self.data.n_train-1, n_examples_display)).type(torch.int)
        self.X_train_small = self.ohe_modifier(self.data.X_train[X_train_small_idcs.to(torch.long)], self.data.src_vocab_size)
        X_val_small_idcs = torch.round(torch.linspace(51, self.data.n_val-1, n_examples_display)).type(torch.int)
        self.X_val_small = self.ohe_modifier(self.data.X_val[X_val_small_idcs.to(torch.long)], self.data.src_vocab_size)

    def train(self, metrics_history=None):
        """
        Description:
            Trains a model.

        Inputs:
            <metrics_history>: Type: <dict>. Holds all training's history evaluation measurements.
        """

        def print_metrics_per_pipeline(pipeline_name):
            print('> '+pipeline_name)
            print('>> '+'Train loss: %f | Val loss: %f'%(metrics_history['loss'][pipeline_name]['train'][-1], metrics_history['loss'][pipeline_name]['val'][-1]))
            print('>> '+'Train BLEU: %f | Val BLEU: %f'%(metrics_history['bleu'][pipeline_name]['train'][-1], metrics_history['bleu'][pipeline_name]['val'][-1]))

        def get_metrics(prediction, ground_truth):

            loss = self.criterion\
            (
                p_distr_pred=prediction,
                p_distr_ground_truth=ground_truth,
                device=self.device,
                ignore_index=self.data.tgt_vocab.get_stoi()['<pad>']
            )
            bleu = torch.tensor\
            (
                metrics.bleu_k\
                (
                    col_pred=dataset.ohe2int(prediction).tolist(), 
                    col_ground_truth=dataset.ohe2int(ground_truth).tolist(),
                    max_k=2,
                    unk=self.data.tgt_vocab.get_stoi()['<unk>'],
                    eos=self.data.tgt_vocab.get_stoi()['<eos>']
                )
            ).to(device=self.device).detach()

            return loss, bleu

        if metrics_history == None:
            metrics_history = \
            {
                metric_name:
                {
                    pipeline_name:
                    {
                        data_subset: [] for data_subset in self.data_subset_names
                    } for pipeline_name in self.pipeline_names
                } for metric_name in self.metric_names
            }

        print('Number of training instances: %d'%(self.data.n_train))
        print('Training Status:')

        if metrics_history['bleu']['id2_pipeline']['val'] == []:
            epochs_bleu_epoch_id2_pipeline_val_hat_prev = float('inf')
        else:
            epochs_bleu_epoch_id2_pipeline_val_hat_prev = metrics_history['bleu']['id2_pipeline']['val'][-1]

        figure_ = plot.figure(metric_names=self.metric_names, pipeline_names=self.pipeline_names)
        initial_epoch = len(metrics_history['loss']['training_pipeline']['val'])
        t_before_training = time()
        for epoch in range(initial_epoch, self.epochs):
            t_i = time()

            losses_minibatch_training_pipeline_train = []
            losses_minibatch_training_pipeline_val = []
            losses_minibatch_id2_pipeline_train = []
            losses_minibatch_id2_pipeline_val = []

            bleus_minibatch_training_pipeline_train = []
            bleus_minibatch_training_pipeline_val = []
            bleus_minibatch_id2_pipeline_train = []
            bleus_minibatch_id2_pipeline_val = []

            for (i, instance_i) in enumerate(self.train_dataloader):

                ## ! Training step initialization: Begin

                current_parameters_are_optimal = False

                X_train_minibatch_i = self.ohe_modifier(instance_i[0], self.data.src_vocab_size)
                Y_train_minibatch_i = self.ohe_modifier(instance_i[1], self.data.tgt_vocab_size)
                Y_bos_train_minibatch_i = self.ohe_modifier(instance_i[2], self.data.tgt_vocab_size)

                ## Resetting gradient tensor variables
                self.optimizer.zero_grad()

                ## ! Training step initialization: End

                ## Forward propagation
                prediction_minibatch_training_pipeline_train = self.model(dec_mode=1, dec_config={'decay': self.decay_scheduler(epoch)}, train=True, X=X_train_minibatch_i, Y=Y_bos_train_minibatch_i)
                prediction_minibatch_id2_pipeline_train = self.model(dec_mode=2, dec_config=None, train=False, X=X_train_minibatch_i, Y=None)

                ## ! Loss and other evaluation metrics: Begin

                ## ! Training pipeline: Begin

                loss_minibatch_training_pipeline_train, \
                bleu_minibatch_training_pipeline_train \
                = get_metrics\
                (
                    prediction=prediction_minibatch_training_pipeline_train, ground_truth=Y_train_minibatch_i
                )

                losses_minibatch_training_pipeline_train.append(loss_minibatch_training_pipeline_train.detach())
                bleus_minibatch_training_pipeline_train.append(bleu_minibatch_training_pipeline_train)

                ## ! Training pipeline: End

                ## ! Prediction pipeline with ID == 2: Begin

                loss_minibatch_id2_pipeline_train, \
                bleu_minibatch_id2_pipeline_train \
                = get_metrics\
                (
                    prediction=prediction_minibatch_id2_pipeline_train, ground_truth=Y_train_minibatch_i
                )

                losses_minibatch_id2_pipeline_train.append(loss_minibatch_id2_pipeline_train.detach())
                bleus_minibatch_id2_pipeline_train.append(bleu_minibatch_id2_pipeline_train)

                ## ! Prediction pipeline with ID == 2: End

                ## ! Loss and other evaluation metrics: End

                ## Backpropagation - Gradients computation
                loss_minibatch_training_pipeline_train.backward()

                self.clip_gradients(grad_clip_val=1, model=self.model)

                ## Gradient Descent - Gradient update
                self.optimizer.step()

            with torch.no_grad():

                for (j, instance_j) in enumerate(self.val_dataloader):

                    ## ! Validation/Test step initialization: Begin

                    X_val_minibatch_j = self.ohe_modifier(instance_j[0], self.data.src_vocab_size)
                    Y_val_minibatch_j = self.ohe_modifier(instance_j[1], self.data.tgt_vocab_size)
                    Y_bos_val_minibatch_j = self.ohe_modifier(instance_j[2], self.data.tgt_vocab_size)

                    ## ! Validation/Test step initialization: End

                    ## Forward propagation
                    prediction_minibatch_training_pipeline_val = self.model(dec_mode=1, dec_config={'decay': self.decay_scheduler(epoch)}, train=False, X=X_val_minibatch_j, Y=Y_bos_val_minibatch_j)
                    prediction_minibatch_id2_pipeline_val = self.model(dec_mode=2, dec_config=None, train=False, X=X_val_minibatch_j, Y=None).detach()

                    ## ! Loss and other evaluation metrics: Begin

                    ## ! Training pipeline: Begin

                    loss_minibatch_training_pipeline_val, \
                    bleu_minibatch_training_pipeline_val \
                    = get_metrics\
                    (
                        prediction=prediction_minibatch_training_pipeline_val, ground_truth=Y_val_minibatch_j
                    )

                    losses_minibatch_training_pipeline_val.append(loss_minibatch_training_pipeline_val)
                    bleus_minibatch_training_pipeline_val.append(bleu_minibatch_training_pipeline_val)

                    ## ! Training pipeline: End

                    ## ! Prediction pipeline with ID == 2: Begin

                    loss_minibatch_id2_pipeline_val, \
                    bleu_minibatch_id2_pipeline_val \
                    = get_metrics\
                    (
                        prediction=prediction_minibatch_id2_pipeline_val, ground_truth=Y_val_minibatch_j
                    )

                    losses_minibatch_id2_pipeline_val.append(loss_minibatch_id2_pipeline_val)
                    bleus_minibatch_id2_pipeline_val.append(bleu_minibatch_id2_pipeline_val)

                    ## ! Prediction pipeline with ID == 2: End

                    ## ! Loss and other evaluation metrics: End

            ## ! Training pipeline: Begin

            loss_epoch_training_pipeline_train_hat = torch.sum(torch.stack(losses_minibatch_training_pipeline_train)) / len(losses_minibatch_training_pipeline_train)
            metrics_history['loss']['training_pipeline']['train'].append(loss_epoch_training_pipeline_train_hat.item())
            bleu_epoch_training_pipeline_train_hat = torch.sum(torch.stack(bleus_minibatch_training_pipeline_train)) / len(bleus_minibatch_training_pipeline_train)
            metrics_history['bleu']['training_pipeline']['train'].append(bleu_epoch_training_pipeline_train_hat.item())

            loss_epoch_training_pipeline_val_hat = torch.sum(torch.stack(losses_minibatch_training_pipeline_val)) / len(losses_minibatch_training_pipeline_val)
            metrics_history['loss']['training_pipeline']['val'].append(loss_epoch_training_pipeline_val_hat.item())
            bleu_epoch_training_pipeline_val_hat = torch.sum(torch.stack(bleus_minibatch_training_pipeline_val)) / len(bleus_minibatch_training_pipeline_val)
            metrics_history['bleu']['training_pipeline']['val'].append(bleu_epoch_training_pipeline_val_hat.item())

            ## ! Training pipeline: End

            ## ! Prediction pipeline with ID == 2: Begin

            loss_epoch_id2_pipeline_train_hat = torch.sum(torch.stack(losses_minibatch_id2_pipeline_train)) / len(losses_minibatch_id2_pipeline_train)
            metrics_history['loss']['id2_pipeline']['train'].append(loss_epoch_id2_pipeline_train_hat.item())
            bleu_epoch_id2_pipeline_train_hat = torch.sum(torch.stack(bleus_minibatch_id2_pipeline_train)) / len(bleus_minibatch_id2_pipeline_train)
            metrics_history['bleu']['id2_pipeline']['train'].append(bleu_epoch_id2_pipeline_train_hat.item())

            loss_epoch_id2_pipeline_val_hat = torch.sum(torch.stack(losses_minibatch_id2_pipeline_val)) / len(losses_minibatch_id2_pipeline_val)
            metrics_history['loss']['id2_pipeline']['val'].append(loss_epoch_id2_pipeline_val_hat.item())
            bleu_epoch_id2_pipeline_val_hat = torch.sum(torch.stack(bleus_minibatch_id2_pipeline_val)) / len(bleus_minibatch_id2_pipeline_val)
            metrics_history['bleu']['id2_pipeline']['val'].append(bleu_epoch_id2_pipeline_val_hat.item())

            ## ! Prediction pipeline with ID == 2: End

            figure_.plot\
            (
                hor_seq=np.arange(0, epoch+1),
                metrics_history = metrics_history
            )

            current_parameters_are_optimal = bleu_epoch_id2_pipeline_val_hat.item() >= max(metrics_history['bleu']['id2_pipeline']['val'])

            self.save_training(epoch, metrics_history=metrics_history, thparams=(self.lr, self.minibatch_size), t_before_training=t_before_training, figure_=figure_, current_parameters_are_optimal=current_parameters_are_optimal)

            t_f = time()
            delta_t = round(t_f-t_i)
            est_next_epoch_time = datetime.datetime.utcfromtimestamp(t_f) + datetime.timedelta(seconds=delta_t)

            print('[Epoch %d @ UTC %s]'%(epoch, datetime.datetime.utcfromtimestamp(t_f).strftime("%H:%M:%S")))

            print_metrics_per_pipeline(pipeline_name='training_pipeline')
            print_metrics_per_pipeline(pipeline_name='id2_pipeline')

            print('Δt: %ds | Δ training pipeline\'s val bleu: %f\nNext epoch @ ~UTC %s'%(delta_t, metrics_history['bleu']['training_pipeline']['val'][-1]-epochs_bleu_epoch_id2_pipeline_val_hat_prev, est_next_epoch_time.strftime("%H:%M:%S")))

            pred_train_small_ohe = self.model(dec_mode=2, dec_config=None, train=False, X=self.X_train_small, Y=None).detach()
            pred_train_small_int = dataset.ohe2int(pred_train_small_ohe)
            pred_train_small_str = dataset.int2str(col_int=pred_train_small_int, vocab=self.data.tgt_vocab, max_steps=self.data.max_steps_tgt)
            X_train_small_str = dataset.ohe2str(self.X_train_small, self.data.src_vocab, self.data.max_steps_src)

            pred_val_small_ohe = self.model(dec_mode=2, dec_config=None, train=False, X=self.X_val_small, Y=None).detach()
            pred_val_small_int = dataset.ohe2int(pred_val_small_ohe)
            pred_val_small_str = dataset.int2str(col_int=pred_val_small_int, vocab=self.data.tgt_vocab, max_steps=self.data.max_steps_tgt)
            X_val_small_str = dataset.ohe2str(self.X_val_small, self.data.src_vocab, self.data.max_steps_src)

            print('Training set examples:')
            for example_idx in range(len(pred_train_small_str)):
                print('%-4s'%(str(example_idx)), end='')
                print('%-40s   '%(X_train_small_str[example_idx]), end='')
                print('%s'%(pred_train_small_str[example_idx]))
            print('Validation set examples:')
            for example_idx in range(len(pred_val_small_str)):
                print('%-4s'%(str(example_idx)), end='')
                print('%-40s   '%(X_val_small_str[example_idx]), end='')
                print('%s'%(pred_val_small_str[example_idx]))

            print()

            epochs_bleu_epoch_id2_pipeline_val_hat_prev = deepcopy(metrics_history['bleu']['id2_pipeline']['val'][-1])

            torch.cuda.empty_cache()

        print('Training completed.')

    def clip_gradients(self, grad_clip_val, model):

        params = [p for p in model.parameters() if p.requires_grad]
        norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
        if norm > grad_clip_val:
            for param in params:
                param.grad[:] *= grad_clip_val / norm

    def save_training(self, epoch, metrics_history, thparams, t_before_training, figure_, current_parameters_are_optimal):

        def get_training_information(epoch, metrics_history, thparams, t_before_training):

            lr, minibatch_size = thparams

            training_information = \
            {
                'model_params': self.model.state_dict(),
                'src_vocab': self.data.src_vocab,
                'src_vocab_size': self.data.src_vocab_size,
                'tgt_vocab': self.data.tgt_vocab,
                'tgt_vocab_size': self.data.tgt_vocab_size,
                'max_steps_src': self.data.max_steps_src,
                'max_steps_tgt': self.data.max_steps_tgt,
                'data_info':
                {
                    'n_train': self.data.n_train,
                    'n_val': self.data.n_val
                },
                'metrics_history': metrics_history,
                'training_hparams':
                {
                    'epoch': epoch,
                    'learning_rate': lr,
                    'minibatch_size': minibatch_size
                },
                'delta_t': time()-t_before_training,
                'shuffle_seed': self.data.shuffle_seed,
                'dataset_name': self.data.dataset_name
            }

            return training_information

        ## Scheduled backup - model thread
        if epoch in self.scheduled_checkpoints:
            training_information = get_training_information(epoch, metrics_history, thparams, t_before_training)
            training_scheduled_backup_path = self.training_dir_path + self.model.name + '_ep' + str(epoch) + self.training_format
            torch.save(training_information, training_scheduled_backup_path)

        ## Latest frequent backups
        if (epoch != 0) and ((epoch % self.bkp_freq) == 0):
            training_information = get_training_information(epoch, metrics_history, thparams, t_before_training)
            live_training_backup_path = self.training_dir_path + self.model.name + '_live_ep' + str(epoch) + self.training_format
            prev_live_training_backup_path = self.training_dir_path + self.model.name + '_live_ep' + str(epoch-self.bkp_freq) + self.training_format
            if os.path.exists(prev_live_training_backup_path):
                os.remove(prev_live_training_backup_path)
            torch.save(training_information, live_training_backup_path)

        if current_parameters_are_optimal:
            training_information = get_training_information(epoch, metrics_history, thparams, t_before_training)
            training_opt_backup_path = self.training_dir_path + self.model.name + '_opt' + self.training_format
            if os.path.exists(training_opt_backup_path):
                os.remove(training_opt_backup_path)
            torch.save(training_information, training_opt_backup_path)

        figure_.fig.savefig(self.training_dir_path + self.model.name + '_live')