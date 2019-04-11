import sys
import pink_utils as pu
import numpy as np
import matplotlib.pyplot as plt

for c, f in enumerate(sys.argv[1:]):

    som = pu.som(f)

    header = som.file_head
    print(header)

    for c in range(header[0]):
        print(c)
        layer = som.get_som(channel=c)

        fig, ax = plt.subplots(1,1, figsize=(header[1:3]))

        ax.imshow(layer, cmap='bwr')
        pu.no_ticks(ax)
        
        for i in range(header[1]+1):
            ax.axvline(i*header[4], color='black')

        for i in range(header[2]+1):
            ax.axhline(i*header[4], color='black')

        ax.set_ylim([0,layer.shape[0]])
        ax.set_xlim([0,layer.shape[1]])

        # ax.set_title(f"Layer {c}")
        fig.tight_layout()
        fig.savefig(f"Layer_{c}_SOM.png")

    plt.close('all')

    neurons = [(y,x) for y in range(header[1]) for x in range(header[2])]

    for count, (y, x) in enumerate(neurons):
        print(f"{y}, {x}")

        fig, axes = plt.subplots(1, header[0], figsize=(10,5))

        for ax, c in zip(axes.flat, range(header[0])):
            img = np.sqrt(som.get_neuron(y=y, x=x, channel=c))
            ax.imshow(img, cmap='bwr')
            pu.no_ticks(ax)

            cen = np.array(img.shape) / 2
            ax.axhline(cen[0], ls='--', color='black')
            ax.axvline(cen[1], ls='--', color='black')

        fig.suptitle(f"{count}) Neuron {y, x}")
        fig.tight_layout(rect=[0,0,0.95,0.95])
        fig.savefig(f"{y}_{x}_neuron.png")
        plt.close('all')

