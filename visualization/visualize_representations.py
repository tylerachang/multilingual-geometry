"""
Code to visualize representations projected onto different axes.
No main function; instead, the following functions should be imported from this
file: visualize_representations_3d, visualize_representations_2d.
See examples in readme and sample Colab notebook.

Must have already extracted representations (extract_representations.py).
For position axes, must have run get_position_representations.py; for POS axes,
must have run get_pos_representations.py. Sample usage:

# Horizontal axes are position LDA axes, vertical axis is a language LDA axis.
from visualize_representations import visualize_representations_3d
visualize_representations_3d(axis0="position:0", axis1="position:1", axis2="lang:0",
                             plot_points="lang:en-zh", color_by="lang", layer=8,
                             points_per_lang=4096, alpha=0.10, s=1.0, azim=30, elev=35,
                             xlabel="Position LDA 0", ylabel="Position LDA 1", zlabel="Language LDA 0",
                             savefile="visualization0.png")

Sample usage for POS LDA:

# Horizontal axes are POS LDA axes, vertical axis is a language LDA axis.
from visualize_representations import visualize_representations_3d
from visualization_utils import get_pos_lda
pos_lda = get_pos_lda(["NOUN", "VERB", "ADJ"], layer=8)
visualize_representations_3d(axis0=pos_lda[:, 0], axis1=pos_lda[:, 1], axis2="lang:0",
                             plot_points="lang:en-zh,pos:NOUN-VERB-ADJ", color_by="pos", layer=8,
                             points_per_lang=4096, alpha=0.10, s=1.0, azim=30, elev=35,
                             xlabel="POS LDA 0", ylabel="POS LDA 1", zlabel="Language LDA 0",
                             savefile="visualization1.png")

"""

import os
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import pickle
from collections import defaultdict

from visualization_utils import get_direction, get_global_mean, get_orthonormal, get_reps_with_token_info, project_reps
from visualization_constants import ALL_POS, POS_REPS_DIR, VISUALIZATION_OUTPUTS

# Function to visualize representations in 3D.
# Axes are orthogonalized in order. Axes can be defined by:
# lang:[dim_i] (language LDA direction)
# position:[dim_i] (position LDA direction)
# langsvd:[langcode]-[dim_i] (SVD direction i for a specific language, must have
# already computed language subspaces using eval_perplexity.py, see readme)
#
# Axes can also be inputted as raw np.ndarrays (e.g. if custom directions are computed using LDA).
# Representations to visualize are defined by plot_points, a comma-separated list of:
# lang:[langcode1]- ... -[langcode_n] (required)
# pos:[pos1]- ... -[pos_n] (optional)
# If pos is omitted, defaults to all parts-of-speech.
# Coloring is determined by color_by: pos, lang, or position:[modulo_value or "absolute"]
def visualize_representations_3d(axis0="", axis1="", axis2="", plot_points="", color_by="", layer=8, points_per_lang=1024,
                                 alpha=0.25, s=0.75, elev=None, azim=None, xlabel="", ylabel="", zlabel="",
                                 savefile="visualization.png", include_legend=True, title="", lims=None, figsize=(4,4), colors=None):
    # Use the global mean as the subspace origin.
    print("Getting mean.")
    axis_origin = get_global_mean(layer)
    # Get axis directions.
    print("Getting axis directions.")
    axis_directions = []
    axis_directions.append(get_direction(axis0, layer))
    axis_directions.append(get_direction(axis1, layer))
    axis_directions.append(get_direction(axis2, layer))
    axis_directions = np.stack(axis_directions, axis=-1)
    # Orthogonalize in order.
    axis_directions[:, 0] = axis_directions[:, 0] / np.linalg.norm(axis_directions[:, 0])
    axis_directions[:, 1] = get_orthonormal(axis_directions[:, 1], axis_directions[:, [0]])
    axis_directions[:, 2] = get_orthonormal(axis_directions[:, 2], axis_directions[:, :2])

    # Load the points to plot.
    print("Getting points.")
    plot_points_split = plot_points.split(",")
    plot_langs = plot_points_split[0].split(":")[1].split("-") # The languages to plot are required.
    plot_pos = [] if len(plot_points_split) <= 1 else plot_points_split[1].split(":")[1].split("-") # Optionally, also filter by POS.
    if color_by == "pos" and len(plot_pos) == 0:
        plot_pos = ALL_POS # Must have run get_pos_representations for all parts-of-speech.
    if len(plot_pos) > 0:
        # Load points by parts-of-speech.
        if color_by == "position":
            print("ERROR: color_by position is not supported when specifying parts-of-speech.")
        reps_dict = defaultdict(list) # Map POS or language (depending on color_by) to the corresponding representations.
        for pos in plot_pos:
            print("Getting points for POS: {}".format(pos))
            with open(os.path.join(POS_REPS_DIR, "{0}_layer{1}_reps.pickle".format(pos, layer)), 'rb') as handle:
                pos_reps = pickle.load(handle)
            for lang in plot_langs:
                # Total points per language should be points_per_lang.
                n_per_pos_lang = points_per_lang // len(plot_pos)
                pos_lang_reps = pos_reps[lang][:n_per_pos_lang]
                if color_by == "pos":
                    reps_dict[pos].append(pos_lang_reps)
                elif color_by == "lang":
                    reps_dict[lang].append(pos_lang_reps)
        for key in reps_dict: # Convert the list of representation arrays to one array per key.
            reps_dict[key] = np.concatenate(reps_dict[key], axis=0)
    else:
        # Otherwise, load by language.
        reps_dict = dict() # Map languages to representations.
        token_info_dict = dict() # Map languages to token info.
        for lang in plot_langs:
            print("Getting points for lang: {}".format(lang))
            reps, token_info_tuples = get_reps_with_token_info(lang, layer, points_per_lang)
            reps_dict[lang] = reps
            token_info_dict[lang] = token_info_tuples

    # Plot point clouds.
    print("Plotting.")
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    if colors is not None:
        ax.set_prop_cycle(color=colors)

    if color_by in ["lang", "pos"]: # Then, reps_dict is already in the correct format.
        for label, reps in reps_dict.items():
            projected_points = project_reps(axis_directions, reps, axis_origin)
            ax.scatter(projected_points[:, 0], projected_points[:, 1], projected_points[:, 2], alpha=alpha, label=label, s=s)
        if include_legend:
            legend = plt.legend()
            for lh in legend.legendHandles:
                lh.set_alpha(1.0)
                lh.set_sizes([100.0])
    elif color_by.split(":")[0] == "position":
        suffix = color_by.split(":")[1]
        # Concatenate all the reps_dicts and token_infos.
        reps = np.concatenate(list(reps_dict.values()), axis=0)
        token_info_tuples = []
        for token_infos in token_info_dict.values():
            token_info_tuples = token_info_tuples + token_infos
        # Get the position information.
        if suffix == "absolute":
            positions = np.asarray([info[1] for info in token_info_tuples])
        else:
            modulo = float(suffix)
            positions = np.asarray([info[1] % modulo for info in token_info_tuples])
        # Plot the points.
        projected_reps = project_reps(axis_directions, reps, axis_origin)
        p = ax.scatter(projected_reps[:, 0], projected_reps[:, 1], projected_reps[:, 2], s=s, c=positions, alpha=alpha)
        if include_legend:
            label = "Position" if suffix == "absolute" else "Position mod {}".format(int(modulo))
            color_bar = fig.colorbar(p, label=label, pad=0.1, shrink=0.75)
            color_bar.set_alpha(1)
            color_bar.draw_all()

    # Optionally set axis limits.
    if lims is not None:
        ax.set_xlim(lims[0], lims[1])
        ax.set_ylim(lims[2], lims[3])
        ax.set_zlim(lims[4], lims[5])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.title(title)
    # Optionally change the view angle:
    ax.view_init(elev=elev, azim=azim)

    if not os.path.isdir(VISUALIZATION_OUTPUTS):
        os.mkdir(VISUALIZATION_OUTPUTS)
    plt.savefig(os.path.join(VISUALIZATION_OUTPUTS, savefile), dpi=300, facecolor='white', bbox_inches='tight')
    print("Done.")
    return True

# Visualize representations in 2D, largely the same as the function above.
# Uses the same syntax as 3D plots.
def visualize_representations_2d(axis0="", axis1="", plot_points="", color_by="", layer=8, points_per_lang=1024,
                                 alpha=0.25, s=0.75, xlabel="", ylabel="", savefile="save.png", include_legend=True,
                                 title="", aspect="equal", lims=None, figsize=(4,4), colors=None):
    # Use the global mean as the subspace origin.
    print("Getting mean.")
    axis_origin = get_global_mean(layer)
    # Get axis directions.
    print("Getting axis directions.")
    axis_directions = []
    axis_directions.append(get_direction(axis0, layer))
    axis_directions.append(get_direction(axis1, layer))
    axis_directions = np.stack(axis_directions, axis=-1)
    # Orthogonalize in order.
    axis_directions[:, 0] = axis_directions[:, 0] / np.linalg.norm(axis_directions[:, 0])
    axis_directions[:, 1] = get_orthonormal(axis_directions[:, 1], axis_directions[:, [0]])

    # Load the points to plot.
    # Same as in 3D.
    print("Getting points.")
    plot_points_split = plot_points.split(",")
    plot_langs = plot_points_split[0].split(":")[1].split("-") # The languages to plot are required.
    plot_pos = [] if len(plot_points_split) <= 1 else plot_points_split[1].split(":")[1].split("-") # Optionally, also filter by POS.
    if color_by == "pos" and len(plot_pos) == 0:
        plot_pos = ALL_POS # Must have run get_pos_representations for all parts-of-speech.
    if len(plot_pos) > 0:
        # Load points by parts-of-speech.
        if color_by == "position":
            print("ERROR: color_by position is not supported when specifying parts-of-speech.")
        reps_dict = defaultdict(list) # Map POS or language (depending on color_by) to the corresponding representations.
        for pos in plot_pos:
            print("Getting points for POS: {}".format(pos))
            with open(os.path.join(POS_REPS_DIR, "{0}_layer{1}_reps.pickle".format(pos, layer)), 'rb') as handle:
                pos_reps = pickle.load(handle)
            for lang in plot_langs:
                # Total points per language should be points_per_lang.
                n_per_pos_lang = points_per_lang // len(plot_pos)
                pos_lang_reps = pos_reps[lang][:n_per_pos_lang]
                if color_by == "pos":
                    reps_dict[pos].append(pos_lang_reps)
                elif color_by == "lang":
                    reps_dict[lang].append(pos_lang_reps)
        for key in reps_dict: # Convert the list of representation arrays to one array per key.
            reps_dict[key] = np.concatenate(reps_dict[key], axis=0)
    else:
        # Otherwise, load by language.
        reps_dict = dict() # Map languages to representations.
        token_info_dict = dict() # Map languages to token info.
        for lang in plot_langs:
            print("Getting points for lang: {}".format(lang))
            reps, token_info_tuples = get_reps_with_token_info(lang, layer, points_per_lang)
            reps_dict[lang] = reps
            token_info_dict[lang] = token_info_tuples

    # Plot point clouds.
    print("Plotting.")
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(aspect=aspect)
    if colors is not None:
        ax.set_prop_cycle(color=colors)

    if color_by in ["lang", "pos"]: # Then, reps_dict is already in the correct format.
        for label, reps in reps_dict.items():
            projected_points = project_reps(axis_directions, reps, axis_origin)
            ax.scatter(projected_points[:, 0], projected_points[:, 1], alpha=alpha, label=label, s=s)
        if include_legend:
            legend = plt.legend()
            for lh in legend.legendHandles:
                lh.set_alpha(1.0)
                lh.set_sizes([100.0])
    elif color_by.split(":")[0] == "position":
        suffix = color_by.split(":")[1]
        # Concatenate all the reps_dicts and token_infos.
        reps = np.concatenate(list(reps_dict.values()), axis=0)
        token_info_tuples = []
        for token_infos in token_info_dict.values():
            token_info_tuples = token_info_tuples + token_infos
        # Get the position information.
        if suffix == "absolute":
            positions = np.asarray([info[1] for info in token_info_tuples])
        else:
            modulo = float(suffix)
            positions = np.asarray([info[1] % modulo for info in token_info_tuples])
        # Plot the points.
        projected_reps = project_reps(axis_directions, reps, axis_origin)
        p = ax.scatter(projected_reps[:, 0], projected_reps[:, 1], s=s, c=positions, alpha=alpha)
        if include_legend:
            label = "Position" if suffix == "absolute" else "Position mod {}".format(int(modulo))
            color_bar = fig.colorbar(p, label=label, pad=0.1, shrink=0.75)
            color_bar.set_alpha(1)
            color_bar.draw_all()

    # Optionally set axis limits.
    if lims is not None:
        ax.set_xlim(lims[0], lims[1])
        ax.set_ylim(lims[2], lims[3])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if not os.path.isdir(VISUALIZATION_OUTPUTS):
        os.mkdir(VISUALIZATION_OUTPUTS)
    plt.savefig(os.path.join(VISUALIZATION_OUTPUTS, savefile), dpi=300, facecolor='white', bbox_inches='tight')
    print("Done.")
    return True
