import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class PlotExplanation(object):
    @staticmethod
    def plot_impacts(data, target_col, ax, title):

        n = len(data)
        ax.set_xlim(-0.5, 0.5)  # set x axis limits
        ax.set_ylim(-1, n)  # set y axis limits
        ax.set_yticks(range(n))  # add 0-n ticks
        ax.set_yticklabels(data[['column', 'word']].apply(lambda x: ', '.join(x), 1))  # add y tick labels

        # define arrows
        arrow_starts = np.repeat(0, n)
        arrow_lengths = data[target_col].values
        # add arrows to plot
        for i, subject in enumerate(data['column']):

            if subject.startswith('l'):
                arrow_color = '#347768'
            elif subject.startswith('r'):
                arrow_color = '#6B273D'

            if arrow_lengths[i] != 0:
                ax.arrow(arrow_starts[i],  # x start point
                         i,  # y start point
                         arrow_lengths[i],  # change in x
                         0,  # change in y
                         head_width=0,  # arrow head width
                         head_length=0,  # arrow head length
                         width=0.4,  # arrow stem width
                         fc=arrow_color,  # arrow fill color
                         ec=arrow_color)  # arrow edge color

        # format plot
        ax.set_title(title)  # add title
        ax.axvline(x=0, color='0.9', ls='--', lw=2, zorder=0)  # add line at x=0
        ax.grid(axis='y', color='0.9')  # add a light grid
        ax.set_xlim(-0.5, 0.5)  # set x axis limits
        ax.set_xlabel('Token impact')  # label the x axis
        sns.despine(left=True, bottom=True, ax=ax)

    @staticmethod
    def plot_landmark(exp, landmark):

        if landmark == 'right':
            target_col = 'score_right_landmark'
        else:
            target_col = 'score_left_landmark'

        data = exp.copy()

        # sort individuals by amount of change, from largest to smallest
        data = data.sort_values(by=target_col, ascending=True) \
            .reset_index(drop=True)

        # initialize a plot
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 6))  # create figure

        if target_col == 'score_right_landmark':
            PlotExplanation.plot_impacts(data[data['column'].str.startswith('l')], target_col, axes[0],
                                         'Original Tokens')
            PlotExplanation.plot_impacts(data[data['column'].str.startswith('r')], target_col, axes[1],
                                         'Augmented Tokens')
        else:
            PlotExplanation.plot_impacts(data[data['column'].str.startswith('r')], target_col, axes[0],
                                         'Original Tokens')
            PlotExplanation.plot_impacts(data[data['column'].str.startswith('l')], target_col, axes[1],
                                         'Augmented Tokens')
            # fig.suptitle('Right Landmark Explanation')
        fig.tight_layout()

    @staticmethod
    def plot(exp):
        data = exp.copy()
        # initialize a plot
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 6))  # create figure
        target_col = 'score_right_landmark'
        # sort individuals by amount of change, from largest to smallest
        data = data.sort_values(by=target_col, ascending=True).reset_index(drop=True)
        PlotExplanation.plot_impacts(data[data['column'].str.startswith('l')], target_col, axes[0], 'Original Tokens')
        PlotExplanation.plot_impacts(data[data['column'].str.startswith('r')], target_col, axes[1], 'Augmented Tokens')
        axes[0].set_ylabel('Right Landmark')
        axes[1].set_ylabel('Right Landmark')

        target_col = 'score_left_landmark'
        PlotExplanation.plot_impacts(data[data['column'].str.startswith('r')], target_col, axes[2], 'Original Tokens')
        PlotExplanation.plot_impacts(data[data['column'].str.startswith('l')], target_col, axes[3], 'Augmented Tokens')
        axes[2].set_ylabel('Left Landmark')
        axes[3].set_ylabel('Left Landmark')

        fig.tight_layout()