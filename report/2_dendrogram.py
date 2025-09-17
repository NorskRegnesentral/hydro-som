# Dendrogram plot
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy
import numpy as np


final_n_clusters = 4

# Load the linkage matrix
link_data = Path(__file__).parent / "links.txt"
links = np.genfromtxt(link_data, dtype=float)

fig = plt.figure(figsize=(10, 8))

# Axis settings
left, bottom, width, height = 0.1, 0.35, 0.8, 0.35
ax1 = fig.add_axes([left, bottom, width, height])

# Customize the colors of the nine clusters
colors=['#BA2F29', '#E9C832', '#8EBA42', '#67ACE6']
hierarchy.set_link_color_palette(colors)

# Plot the dendrogram using the built-in scipy function
# Ideally, we could use maxclust directly, like we do with fcluster:
#   hierarchy.fcluster(links, i, 'maxclust')
# However, this is not supported in dendrogram, so we need to find
# the appropriate distance threshold to get the desired number of clusters
max_d = 0.7
for _ in range(20):
    # Don't plot yet, this is just to get the number of clusters
    dendrogram = hierarchy.dendrogram(
        links,
        leaf_rotation=90,
        leaf_font_size=0,
        color_threshold=max_d,
        above_threshold_color='grey',
        no_plot=True
    )
    d_clusts = len(set(dendrogram['leaves_color_list']))
    if d_clusts == final_n_clusters:
        break
    if d_clusts < final_n_clusters:
        max_d *= 0.8
    if d_clusts > final_n_clusters:
        max_d *= 1.2
else:
    # This can happen if e.g. two clusters are very close together
    print(f"Warning: Could not find a distance threshold to get {final_n_clusters} clusters")

# Main dendrogram plot
print(f"Using distance threshold {max_d:.3f} to get {d_clusts} clusters")
dendrogram = hierarchy.dendrogram(
        links,
        leaf_rotation=90,
        leaf_font_size=0,
        color_threshold=max_d,
        above_threshold_color='grey',
)
hierarchy.set_link_color_palette(None)
ax1.axhline(y=max_d, linestyle='-.', color='k', lw=1.25) 
ax1.set_ylabel('Linkage distance', fontsize=13)
ax1.set_yticklabels([0, 5, 10, 15, 20, 25, 30, 35], fontsize=10)
ax1.set_xticks([])
ax1.spines['top'].set_linewidth(1.25)
ax1.spines['bottom'].set_linewidth(1.25)
ax1.spines['left'].set_linewidth(1.25)
ax1.spines['right'].set_linewidth(1.25)
# ax1.text(150, 20.5, '$Phenon$' + ' ' + '$Line$', fontsize=12)


# The bottom figure showing the cluster names
left, bottom, width, height = 0.1, 0.15, 0.8, 0.2
ax2 = fig.add_axes([left, bottom, width, height])
ax2.set_xlim(0, (len(links) + 1) * 10)
ax2.set_ylim(0, 2)


# Dot line to split each cluster 
unique, counts = np.unique(dendrogram['leaves_color_list'], return_counts=True)
cluster_size_dict = dict(zip(unique, counts))
cluster_size = [
    cluster_size_dict[clr]
    for clr in dict.fromkeys(dendrogram['leaves_color_list'])
]
boundaries = np.cumsum(cluster_size) * 10
for i in range(final_n_clusters - 1):
    ax2.plot([boundaries[i], boundaries[i]], [1.5, 2.0], linestyle='--', color='k', lw=1.5) 

ax2.set_xticks([])
ax2.set_yticks([])
    
ax2.spines['top'].set_color('none')
ax2.spines['bottom'].set_color('none')
ax2.spines['left'].set_color('none')
ax2.spines['right'].set_color('none')

# Add cluster labels for arbitrary number of clusters
extended_boundaries = np.concatenate([[0], boundaries])
for i in range(final_n_clusters):
    left_bound = extended_boundaries[i]
    right_bound = extended_boundaries[i + 1]
    center_x = left_bound + (right_bound - left_bound) / 2
    ax2.text(center_x, 1.5, f'C{i+1} \nN={cluster_size[i]}', ha='center', va='bottom', fontsize=12)

plt.show()