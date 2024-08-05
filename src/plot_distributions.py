import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('TkAgg')
sns.set_style("dark")

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 14,
    })

bins = np.linspace(-0.05, 1.05, num=12, endpoint=True)
colors=["#93c47d","#5a5add", "#dd5a5a"]
edge_colors=["#76b45a","#3030d4", "#d43030"]

cij_noprunin_epoch_0 = np.load("../dump/cij_nopruning_epoch_first.npy")
cij_noprunin_epoch_190 = np.load("../dump/cij_nopruning_epoch_best.npy")
cij_prunin_epoch_333 = np.load("../dump/cij_pruning_epoch_last.npy")

classes_to_show = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

fig, axes = plt.subplots(nrows=len(classes_to_show), ncols=1, sharex=True, gridspec_kw={'hspace': 0.1})

for i, class_id in enumerate(classes_to_show): 
    axes[i].set_ylabel("{}".format(str(class_id)))
    axes[i].set_yscale('log')
    axes[i].hist(cij_noprunin_epoch_190[class_id,class_id], bins=bins, color = colors[2], alpha=1, label="Epoch 190",edgecolor=edge_colors[2])
    axes[i].hist(cij_noprunin_epoch_0[class_id, class_id], bins=bins, color = colors[1], alpha=1, label="Epoch 1",edgecolor=edge_colors[1])
    axes[i].set_xticks([])
    axes[i].set_yticks([1000])
axes[-1].set_xticks(bins)

handles, labels = plt.handles, labels = axes[0].get_legend_handles_labels()
order = [1, 0]
fig.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc="upper right", facecolor="white")
fig.tight_layout()
plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.savefig("../figures/cij_distributions_unpruned.pdf", format="pdf", dpi=1200, bbox_inches="tight")
plt.show()
plt.close()

fig, axes = plt.subplots(nrows=len(classes_to_show), ncols=1, sharex=True, gridspec_kw={'hspace': 0.1})

for i, class_id in enumerate(classes_to_show):  
    axes[i].set_ylabel("{}".format(str(class_id)))
    axes[i].set_yscale('log')
    axes[i].hist(cij_noprunin_epoch_190[class_id,class_id], bins=bins, color = colors[2], alpha=1, label="CapsNet+Q",edgecolor=edge_colors[2])
    axes[i].hist(cij_prunin_epoch_333[class_id, class_id], bins=bins, color = colors[0], alpha=1, label="CapsNet+REM",edgecolor=edge_colors[0])
    axes[i].set_xticks([])
    axes[i].set_yticks([1000])

handles, labels = plt.handles, labels = axes[0].get_legend_handles_labels()
order = [1, 0]
fig.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc="upper right",facecolor="white")
fig.tight_layout()
plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.savefig("../figures/cij_distributions_prunedvsunpruned.pdf", format="pdf", dpi=1200, bbox_inches="tight")
plt.show()
plt.close()