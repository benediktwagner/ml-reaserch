import numpy as np
import matplotlib.pyplot as plt

# Barplot
def draw_ltn_operators(a, c, labels, b=None, title=None, save_as=None):
    #plt.style.use('classic')
    idx = np.arange(len(labels))
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    width = 0.35
    
    if b is not None:
        ax.bar(idx, a, width=0.2, label='a', color='#c6c6c6', alpha=0.5, edgecolor='#c6c6c6', hatch='///')
        ax.bar(idx + 0.2, b, width=0.2, label='b', color='#c6c6c6', alpha=0.5, edgecolor='#c6c6c6', hatch='//')
        ax.bar(idx + 0.4, c, width=0.2, label='c', color='#0070c0', alpha=0.5, edgecolor='#0070c0')
        max_value = max(np.concatenate((a,b,c), axis=0))
    else:
        ax.bar(idx, a, width=0.2, label='a', color='#c6c6c6', alpha=0.5, edgecolor='#c6c6c6', hatch='///')
        #ax.bar(idx + 0.2, b, width=0.2, label='b', color='#c6c6c6', alpha=0.5, edgecolor='#c6c6c6', hatch='///')
        ax.bar(idx + 0.2, c, width=0.2, label='c', color='#0070c0', alpha=0.5, edgecolor='#0070c0')
        max_value = max(np.concatenate((a,c), axis=0))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(color='#c6c6c6', linestyle='-', linewidth=0.1, alpha=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', colors='#0070c0')
    ax.tick_params(axis='y', colors='#0070c0')
    plt.ylim((0, max_value+max_value*0.25))
    plt.xticks(idx + width / 2, labels)
    plt.legend(loc='best')
    plt.xlabel(title)
    ax.xaxis.label.set_color('#0070c0')
    #plt.title(title)
    #plt.show()
    if save_as != None:
        fig.savefig(save_as)
    return None


# Reduces a list of lists to a list
def flat_list(list_of_lists):
    return [i for row in list_of_lists for i in row]