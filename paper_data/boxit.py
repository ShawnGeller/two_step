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
    meanlineprops = dict(linestyle='-', linewidth=2.5, color='orange')
    boxplot = ax.bxp(dics, showfliers=False, showmeans=True, meanprops=meanlineprops, meanline=True)

    # Grab the relevant Line2D instances from the boxplot dictionary
    iqr = boxplot['boxes'][0]
    caps = boxplot['caps']
    med = boxplot['medians'][0]

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
    capbottom = caps[0].get_ydata()[0]
    captop = caps[1].get_ydata()[0]

    # Make some labels on the figure using the values derived above
    ax.text(xlabel, median,
            'Median = {:6.3g}'.format(median), va='center')
    ax.text(xlabel, pc25,
            '25th percentile = {:6.3g}'.format(pc25), va='center')
    ax.text(xlabel, pc75,
            '75th percentile = {:6.3g}'.format(pc75), va='center')
    ax.text(xlabel, capbottom,
            'Bottom cap = {:6.3g}'.format(capbottom), va='center')
    ax.text(xlabel, captop,
            'Top cap = {:6.3g}'.format(captop), va='center')



def main(args):
    fn = args.input_file
    with open(fn, 'r') as f:
        lns = f.readlines()
    lns = lns[1:]
    lns = [list(map(float, ln.split(",")))[1:] for ln in lns]
    fig = plt.figure()
    ax = fig.gca()
    do_box(ax, lns)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", metavar="input-file", type=str)
    args = parser.parse_args()
    main(args)
