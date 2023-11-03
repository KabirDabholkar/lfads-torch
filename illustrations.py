import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


def cuboid_data2(o, size=(1, 1, 1)):
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float)
    for i in range(3):
        X[:, :, i] *= size[i]
    X += np.array(o)
    return X


def plotCubeAt2(positions, sizes=None, colors=None, **kwargs):
    if not isinstance(colors, (list, np.ndarray)): colors = ["C0"] * len(positions)
    if not isinstance(sizes, (list, np.ndarray)): sizes = [(1, 1, 1)] * len(positions)
    g = []
    for p, s, c in zip(positions, sizes, colors):
        g.append(cuboid_data2(p, size=s))
    return Poly3DCollection(np.concatenate(g),
                            facecolors=np.repeat(colors, 6), **kwargs)


def generate_plots_loop():

    positions = [(0,0,0),(0,0,1),(0,-0.5,0),(0,-0.5,1)]
    sizes = [(2,1.2,1),(2,1.2,0.5),(2,0.5,1),(2,0.5,0.5)]
    colors = ["crimson","crimson","crimson","C2"]

    for i in range(1,len(positions)+1):
        # print(list(zip(*list(zip(positions,sizes,colors))[:i])))
        generate_plot(
            # *zip(*list(zip(positions,sizes,colors))[:i]),
            positions[:i],
            sizes[:i],
            colors[:i],
            filename=f'nlb_breakup_{i}.png',
            title='NLB co-smoothing'
        )

    #######

    sizes[1] = (2,0.2,0.5)

    for i in range(1,len(positions)+1):
        # print(list(zip(*list(zip(positions,sizes,colors))[:i])))
        generate_plot(
            # *zip(*list(zip(positions,sizes,colors))[:i]),
            positions[:i],
            sizes[:i],
            colors[:i],
            filename=f'fewshot_breakup_{i}.png',
            title='few-shot co-smoothing'
        )

def generate_plot(positions, sizes, colors,filename='nlb_breakup.png',title=None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    if title:
        ax.set_title(title)

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')

    pc = plotCubeAt2(positions, sizes, colors=colors, edgecolor="k",alpha=0.3)
    ax.add_collection3d(pc)
    ax.set_xlabel('time')
    ax.set_ylabel('trials')
    ax.set_zlabel('neurons')
    ax.set_xlim([-.2, 2.2])
    ax.set_ylim([-.7, 1.2])
    ax.set_zlim([-.2, 1.7])

    # ax.set_zlim([-3, 9])
    ax.set_aspect('equal')
    #ax.axis(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    fig.tight_layout()
    fig.savefig(filename,dpi=200)

generate_plots_loop()