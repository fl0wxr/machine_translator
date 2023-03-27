import math
import torch
import dataset


def categ_cross_entropy(p_distr_pred, p_distr_ground_truth, device, ignore_index=None):
    '''
    Inputs:
        <p_distr_pred>: Type: <torch.Tensor>. Shape: (n_examples, max_steps_tgt, tgt_vocab_size). Estimation of probability distribution.
        <p_distr_ground_truth>: Type: <torch.Tensor>. Shape: (n_examples, max_steps_tgt, tgt_vocab_size). Underlying probability distribution.
        <ignore_index>: Type: <int> or <NoneType>. If <ignore_index> is an integer then the loss function's computation will exclude the class with index <ignore_index>. In case <ignore_index> is set to None then the loss function will be computed normally, i.e. over all indices of <p_distr_pred> and <p_distr_ground_truth>. By default <ignore_index> is set to None.

    Outputs:
        <loss_>: Type <torch.Tensor>. Shape: (). The loss function. If the <grad_fn> attribute is None then there is an issue.
    '''

    numerical_stabilizer = 10**-20

    mask = torch.ones(p_distr_ground_truth.shape, dtype=torch.float32, device=device)
    if ignore_index != None:
        mask[:,:,ignore_index] = 0

    scaler = 400. / mask.sum()
    loss_ = - scaler * torch.sum((mask * p_distr_ground_truth) * torch.log(p_distr_pred+numerical_stabilizer), axis=(0, 1, 2))

    return loss_

def bleu_k(col_pred, col_ground_truth, max_k, unk, eos):
    '''
    Description:
        BLEU function defined with respect to 1-grams, ..., max_k-grams. <d0> is defined to either be <int> or <str>. The inputs have to be trimmed and as a result they shouldn't contain the end of sequence or padding tokens.

    Examples:
        bleu = bleu_k(col_pred=pred_str, col_ground_truth=tgt_str, max_k=2, unk='<unk>', eos='<eos>')

    Inputs:
        <col_pred>: Type: <list[<list[<d0>]>]>. The outer list's length is n_examples. The inner list's length can vary depending on the corresponding number of steps.
        <col_ground_truth>: Type: <list[<list[<d0>]>]>. The outer list's length is n_examples. The inner list's length can vary depending on the corresponding number of steps.
        <max_k>: Type: <int>. Maximum number of gram mergings.
        <unk>: Type: <d0>. The unknown token in integer format.
        <eos>: Type: <d0>. The end-of-sequence token in integer format.

    Outputs:
        <bleu_k_value>: Type <float>.
    '''

    assert len(col_pred) == len(col_ground_truth), 'E: Input collections do not share a common multitude of examples.'

    n_examples = len(col_pred)

    ## Trimming
    col_pred = dataset.trim_sequence(seq=col_pred, eos=eos)
    col_ground_truth = dataset.trim_sequence(seq=col_ground_truth, eos=eos)

    ## Make the unkown token differ in the sequences to prevent it from contributing to the score. There are, of course, many ways to achieve this.
    col_ground_truth = [[token if (token != unk) else -1 for token in example] for example in col_ground_truth]

    col_pred_kgrams = []
    col_ground_truth_kgrams = []

    bleu_ = [None for i in range(n_examples)]

    for k_idx in range(max_k):
        k = k_idx+1

        col_pred_kgrams.append(dataset.single_gram_to_k_grams(col_pred, k))
        col_ground_truth_kgrams.append(dataset.single_gram_to_k_grams(col_ground_truth, k))

    for i in range(n_examples):

        product_of_weighted_precisions = 1
        n_pred = len(col_pred[i])
        n_ground_truth = len(col_ground_truth[i])

        if min(n_pred, n_ground_truth) == 0:
            if n_ground_truth == 0:
                bleu_[i] = 1
            else: ## equivalent to `elif (n_ground_truth == 1) and (n_pred==0):`
                bleu_[i] = 0
        else:
            brevity_penalty = min(1.0, math.exp(1 - n_ground_truth/n_pred))

            for k_idx in range(max_k):
                k = k_idx+1

                kgram_instability = k > min([min(len(col_pred[i]), len(col_ground_truth[i])) for i in range(n_examples)])
                if kgram_instability:
                    break

                kgrams_precision = 0. ## Given an example i, for each member of <valid_kgrams>, compute the number of occurences inside <col_ground_truth_kgrams[k_idx][i]> and inside <col_pred_kgrams[k_idx][i]>, take their minimum and add that value. Finally divide by the number of predicted k-grams. The rest is obvious.

                n_kgrams_pred = len(col_pred_kgrams[k_idx][i])
                n_kgrams_ground_truth = len(col_ground_truth_kgrams[k_idx][i])

                valid_kgrams = set() ## kgrams that exist inside <col_ground_truth_kgrams[k_idx][i]>.

                ## Find the common tokens
                for seq_pred_idx in range(n_kgrams_pred):
                    if col_pred_kgrams[k_idx][i][seq_pred_idx] in col_ground_truth_kgrams[k_idx][i]:
                        valid_kgrams.add(col_pred_kgrams[k_idx][i][seq_pred_idx])

                for valid_kgram in valid_kgrams:
                    n_occurences_pred = col_pred_kgrams[k_idx][i].count(valid_kgram)
                    n_occurences_ground_truth = col_ground_truth_kgrams[k_idx][i].count(valid_kgram)
                    kgrams_precision += min((n_occurences_pred, n_occurences_ground_truth))

                kgrams_precision /= n_kgrams_pred

                product_of_weighted_precisions *= kgrams_precision**((1/2)**k)

            bleu_[i] = brevity_penalty * product_of_weighted_precisions

    bleu = sum(bleu_)/n_examples

    return bleu