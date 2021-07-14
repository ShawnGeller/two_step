import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def do_box(ax, lsts):
    dics = []
    for lst in lsts:
        lower_ref, upper_ref, lower_edge_1s, upper_edge_1s, lower_edge_2s, upper_edge_2s = lst
        dic = {
            "med": lower_ref,
            "q1": lower_edge_1s,
            "q3": upper_edge_1s,
            "whislo": lower_edge_2s,
            "whishi": upper_edge_2s,
            "mean": upper_ref,
        }
        dics.append(dic)
    meanlineprops = dict(linestyle='-', linewidth=1, color='orange')
    boxplot = ax.bxp(dics, showfliers=False, showmeans=True, meanprops=meanlineprops, meanline=True)

    for i in range(len(boxplot['boxes'])):
        # Grab the relevant Line2D instances from the boxplot dictionary
        iqr = boxplot['boxes'][i]
        caps = boxplot['caps']
        med = boxplot['medians'][i]

        # The x position of the median line
        xpos = med.get_xdata()

        # Lets make the text have a horizontal offset which is some
        # fraction of the width of the box
        xoff = 0.10 * (xpos[1] - xpos[0])

        # The x position of the labels
        xlabel = xpos[1] + xoff

        # The median is the y-position of the median line
        median = med.get_ydata()[1]

        # The 25th and 75th percentiles are found from the
        # top and bottom (max and min) of the box
        pc25 = iqr.get_ydata().min()
        pc75 = iqr.get_ydata().max()

        # The caps give the vertical position of the ends of the whiskers
        capbottom = caps[2*i].get_ydata()[0]
        captop = caps[2*i+1].get_ydata()[0]

        # Make some labels on the figure using the values derived above
        ax.text(xlabel, pc75,
                r'1 $\sigma$ = {:.1e}'.format(pc75), va='center')
        ax.text(xlabel, captop,
                r'2 $\sigma$ = {:.1e}'.format(captop), va='center')
        ax.text(xlabel, median,
                'ref = {:.1e}'.format(median), va='center')
        if ~np.isclose(median, pc25):
            ax.text(xlabel, pc25,
                    r'1 $\sigma$ = {:.1e}'.format(pc25), va='center')
        if ~np.isclose(pc25, capbottom):
            ax.text(xlabel, capbottom,
                    r'2 $\sigma$ = {:.1e}'.format(capbottom), va='center')
    ax.set_xticks([np.mean(med.get_xdata()) for med in boxplot['medians']])
    ax.set_xticklabels([r"$S_+$", r"$S_-$"])
    ax.set_yticks([])
    ax.set_title("490 GHz Infidelities")



# def main(args):
#     fn = args.input_file
def main(fn):
    with open(fn, 'r') as f:
        lns = f.readlines()
    lns = lns[1:]
    lns = [list(map(float, ln.split(",")))[1:] for ln in lns]
    fig = plt.figure()
    ax = fig.gca()
    for d in ["left", "top", "bottom", "right"]:
        ax.spines[d].set_visible(False)
    # ax.axis('off')
    do_box(ax, lns)
    # plt.savefig("490ghzboxplot.svg")
    plt.show()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("input_file", metavar="input-file", type=str)
    # args = parser.parse_args()
    # main(args)
    main("/home/shawn/redundant_fidelity/paper_data/490ghzoutput.csv")
