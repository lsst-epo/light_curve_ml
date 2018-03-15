from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Data to plot.

    # max features
    x = np.arange(5)
    y = np.arange(5)
    z = np.random.rand(5, 5)

    cs = plt.contourf(x, y, z, corner_mask=True)

    fontP = FontProperties()
    fontP.set_size('small')

    nm, lbl = cs.legend_elements()
    plt.legend(nm, lbl, title=None, prop=fontP, loc='center left',
               bbox_to_anchor=(1, 0.5))
    plt.contour(cs, colors='k')
    plt.title('stuff')

    # Plot grid
    plt.grid(c='k', ls='-', alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()
