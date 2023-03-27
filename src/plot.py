import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import Counter
from copy import deepcopy
from ing_theme_matplotlib import mpl_style


class figure:

    def __init__(self, metric_names, pipeline_names):

        self.metric_names = metric_names
        self.pipeline_names = pipeline_names

        ## 2 side by side axes
        asp_ratio = 0.7# 0.4375 
        l = 1400# 1400
        h = asp_ratio * l
        self.fig = plt.figure(figsize=(l/96, h/96), dpi=96)
        mpl_style(dark=True)

        self.ax = \
        [
            [
                None
                for metric_idx in range(len(self.metric_names))
            ] for pipeline_idx in range(len(self.pipeline_names))
        ]

        ax_idx = 0
        for pipeline_idx in range(len(self.pipeline_names)):
            for metric_idx in range(len(self.metric_names)):
                ax_idx += 1
                self.ax[pipeline_idx][metric_idx] = self.fig.add_subplot\
                (
                    len(self.pipeline_names)*100 + len(self.metric_names)*10 + ax_idx
                )

    def plot(self, hor_seq, metrics_history):

        def plot_(hor_seq, ax_idx, train_seq, val_seq, ver_name, title_name):

            pipeline_idx, metric_idx = ax_idx

            ## Clearing previous frame
            self.ax[metric_idx][pipeline_idx].clear()

            ## To show only integer numbers on the x-axis
            self.ax[metric_idx][pipeline_idx].xaxis.set_major_locator(MaxNLocator(integer=True))

            self.ax[metric_idx][pipeline_idx].set_xlabel('epoch')
            self.ax[metric_idx][pipeline_idx].set_ylabel(ver_name)

            self.ax[metric_idx][pipeline_idx].plot(hor_seq, train_seq, color='red', label='etr %s'%(ver_name))
            self.ax[metric_idx][pipeline_idx].plot(hor_seq, val_seq, color='cyan', label='val %s'%(ver_name))

            self.ax[metric_idx][pipeline_idx].set_title(title_name)

            self.ax[metric_idx][pipeline_idx].legend()

            # self.ax[metric_idx][pipeline_idx].grid(visible=True, color='gray', alpha=0.5)

        for (pipeline_idx, pipeline_name) in enumerate(self.pipeline_names):
            pipeline_name_ = deepcopy(pipeline_name)
            for (metric_idx, metric_name) in enumerate(self.metric_names):
                plot_\
                (
                    hor_seq=hor_seq,
                    ax_idx=(pipeline_idx, metric_idx),
                    train_seq=metrics_history[metric_name][pipeline_name]['train'],
                    val_seq=metrics_history[metric_name][pipeline_name]['val'],
                    ver_name=self.metric_names[metric_idx],
                    title_name=pipeline_name_,
                )
                pipeline_name_ = ''


def plot_sentence_size(data_pair, image_name):

    mpl_style(dark=True)

    src, tgt = data_pair

    src_sentence_len = [len(sentence) for sentence in src]
    tgt_sentence_len = [len(sentence) for sentence in tgt]

    src_sentence_len_counted = Counter(src_sentence_len)
    src_sentence_len_counted_sorted = sorted(src_sentence_len_counted.items(), key=lambda x: x[0], reverse=False)

    tgt_sentence_len_counted = Counter(tgt_sentence_len)
    tgt_sentence_len_counted_sorted = sorted(tgt_sentence_len_counted.items(), key=lambda x: x[0], reverse=False)

    src_possible_sentence_len = [element[0] for element in src_sentence_len_counted_sorted]
    src_max_possible_sentence_len = max(src_possible_sentence_len)

    tgt_possible_sentence_len = [element[0] for element in tgt_sentence_len_counted_sorted]
    tgt_max_possible_sentence_len = max(tgt_possible_sentence_len)

    max_possible_sentence_len = max([src_max_possible_sentence_len, tgt_max_possible_sentence_len])

    range_possible_sentence_len = list(range(0, max_possible_sentence_len+1))
    src_sentence_len_counted_sorted_extended = [0 for sentence_len in range_possible_sentence_len]
    tgt_sentence_len_counted_sorted_extended = [0 for sentence_len in range_possible_sentence_len]

    for sentence_len in range_possible_sentence_len:

        if sentence_len in src_sentence_len_counted.keys():
            src_sentence_len_counted_sorted_extended[sentence_len] = src_sentence_len_counted[sentence_len]

        if sentence_len in tgt_sentence_len_counted.keys():
            tgt_sentence_len_counted_sorted_extended[sentence_len] = tgt_sentence_len_counted[sentence_len]

    asp_ratio = 0.8
    l = 700
    h = asp_ratio * l
    plt.figure(figsize=(l/96, h/96), dpi=96)

    plt.plot(range_possible_sentence_len, src_sentence_len_counted_sorted_extended, color='orange', label='source')
    plt.plot(range_possible_sentence_len, tgt_sentence_len_counted_sorted_extended, color='lime', label='target')
    # plt.title('Frequency Graph')
    plt.xticks(range_possible_sentence_len[::5])
    plt.xlabel('Token Count')
    plt.ylabel('# of Instances')
    plt.legend()
    # plt.grid(visible=True, color='gray', alpha=0.5)

    plt.savefig('../datasets/'+image_name)

def plot_frequency_curves(freqs_pair, image_name):

    mpl_style(dark=True)

    asp_ratio = 0.8
    l = 700
    h = asp_ratio * l
    plt.figure(figsize=(l/96, h/96), dpi=96)

    src_freqs, tgt_freqs = freqs_pair

    src_word_keys = list(range(len(src_freqs)))
    plt.plot(src_word_keys, src_freqs, color='orange', label='source')
    tgt_word_keys = list(range(len(tgt_freqs)))
    plt.plot(tgt_word_keys, tgt_freqs, color='lime', label='target')

    # plt.title('Frequency Graph')
    plt.xlabel('Token Index')
    plt.ylabel('Frequency')
    # plt.grid(visible=True, color='gray', alpha=0.5)
    plt.semilogx()
    plt.semilogy()

    plt.legend()

    plt.savefig('../datasets/'+image_name)