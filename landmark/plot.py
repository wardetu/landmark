import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class PlotExplanation(object):

    def __init__(self, expl):
        self.expl = expl
        expl_impacts = expl.get_expl_impacts()

        # expl_impacts['provenance'] = expl_impacts['column'].apply(lambda x: 'l' if x.startswith('left') else 'r')
        # expl_impacts['ix'] = expl_impacts['provenance'] + "_" + expl_impacts['word']
        expl_impacts['ix'] = expl_impacts['column'] + "_" + expl_impacts['word']
        expl_impacts = expl_impacts[['ix', 'score_left_landmark', 'score_right_landmark']]
        self.expl_impacts = expl_impacts

    @staticmethod
    def plot_impacts(data, target_col, ax, title):

        n = len(data)
        ax.set_xlim(-0.5, 0.5)  # set x axis limits
        ax.set_ylim(-1, n)  # set y axis limits
        ax.set_yticks(range(n))  # add 0-n ticks
        ax.set_yticklabels(data['ix'])  # add y tick labels

        # define arrows
        arrow_starts = np.repeat(0, n)
        arrow_lengths = data[target_col].values

        # add arrows to plot
        for i, subject in enumerate(data['ix']):

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
        if title == 'Original Tokens':
            ax.set_ylabel('Token')  # label the y axis
        sns.despine(left=True, bottom=True, ax=ax)

    def plot_landmark(self, landmark):

        if landmark == 'right':
            target_col = 'score_right_landmark'
        else:
            target_col = 'score_left_landmark'

        data = self.expl_impacts.copy()

        # sort individuals by amount of change, from largest to smallest
        data = data.sort_values(by=target_col, ascending=True) \
            .reset_index(drop=True)

        # initialize a plot
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 6))  # create figure

        if target_col == 'score_right_landmark':
            PlotExplanation.plot_impacts(data[data['ix'].str.startswith('l')], target_col, axes[0], 'Original Tokens')
            PlotExplanation.plot_impacts(data[data['ix'].str.startswith('r')], target_col, axes[1], 'Augmented Tokens')
        else:
            PlotExplanation.plot_impacts(data[data['ix'].str.startswith('r')], target_col, axes[0], 'Original Tokens')
            PlotExplanation.plot_impacts(data[data['ix'].str.startswith('l')], target_col, axes[1], 'Augmented Tokens')
            # fig.suptitle('Right Landmark Explanation')
        fig.tight_layout()

    def plot(self):

        print(self.expl.get_record())
        self.plot_landmark('right')
        self.plot_landmark('left')