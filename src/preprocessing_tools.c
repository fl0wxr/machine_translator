#include <stdio.h>
#include <stdlib.h>
#include <string.h>


int arstr2num(char ***col_seq, int col_seq_length, char **vocab_itos, int vocab_itos_length, int **enc_col_seq)
{
    int word_exists, col_seq_idx, seq_word_idx, vocab_word_idx, unk_token_int = -1;
    char token_str[1000], *end_of_sequence_token_str = "<eos>", *unk_token_str = "<unk>";

    for (vocab_word_idx = 0; vocab_word_idx < vocab_itos_length; ++vocab_word_idx)
    {
        if (strcmp(*(vocab_itos+vocab_word_idx), unk_token_str) == 0)
        {
            unk_token_int = vocab_word_idx;
        }
    }
    if (unk_token_int == -1)
    {
        printf("E: Unknown vocabulary value.\n");
        return 1;
    }

    for (col_seq_idx = 0; col_seq_idx < col_seq_length; ++col_seq_idx)
    {
        seq_word_idx = 0;
        do
        {
            word_exists = 0;

            strcpy(token_str, *(*(col_seq+col_seq_idx)+seq_word_idx));
            for (vocab_word_idx = 0; vocab_word_idx < vocab_itos_length; ++vocab_word_idx)
            {
                if (strcmp(token_str, *(vocab_itos+vocab_word_idx)) == 0)
                {
                    *(*(enc_col_seq+col_seq_idx)+seq_word_idx) = vocab_word_idx;
                    word_exists = 1;
                    break;
                }
            }
            if (!word_exists)
            {
                *(*(enc_col_seq+col_seq_idx)+seq_word_idx) = unk_token_int;
            }

            seq_word_idx++;
        }
        while (strcmp(token_str, end_of_sequence_token_str) != 0);
    }

    return 0;
}

int pad_or_trim(int **enc_col, int enc_col_length, int eos_int, int pad_int, int t_bound, int **enc_col_)
{
    int col_seq_idx, token_int, seq_length;

    for (col_seq_idx = 0; col_seq_idx < enc_col_length; col_seq_idx++)
    {
        // Compute each sequence's length.
        seq_length = 0;
        do
        {
            token_int = *(*(enc_col+col_seq_idx)+seq_length);
            seq_length++;
        }
        while (token_int != eos_int);

        if (seq_length < t_bound)
        {
            // Copy the input sequence to the initial segment of the output sequence
            for (token_int = 0; token_int < seq_length; token_int++)
            {
                *(*(enc_col_+col_seq_idx)+token_int) = *(*(enc_col+col_seq_idx)+token_int);
            }
            // Pad
            for (token_int = seq_length; token_int < t_bound; token_int++)
            {
                *(*(enc_col_+col_seq_idx)+token_int) = pad_int;
            }
        }
        else
        {
            // Trim
            for (token_int = 0; token_int < t_bound-1; token_int++)
            {
                *(*(enc_col_+col_seq_idx)+token_int) = *(*(enc_col+col_seq_idx)+token_int);
            }
            *(*(enc_col_+col_seq_idx)+(t_bound-1)) = eos_int;
        }
    }

    return 0;
}