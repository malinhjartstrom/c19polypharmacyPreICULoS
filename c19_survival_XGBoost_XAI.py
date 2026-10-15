# -------------------------------------------------------------------
# Import required packages and setup environment
# -------------------------------------------------------------------
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, statistics
from shap.utils import safe_isinstance
from shap.utils._exceptions import DimensionError
from shap import Explanation
from shap.plots import colors
from shap.plots._labels import labels
from shap.plots._utils import (
    convert_color,
    convert_ordering,
    get_sort_order,
    merge_nodes,
    sort_inds,
)
from typing import Literal
import scipy
from matplotlib import colormaps
from matplotlib.patches import Ellipse
import scipy.stats as st
from scipy.stats import t
from matplotlib.figure import Figure
from typing import Any

import shap
from loess import loess_1d
from sksurv.metrics import concordance_index_ipcw, concordance_index_censored
from sksurv.util import Surv
from org import excel, write_to_file, pdf, save_as_svg # Local functions for organizing the output
from datetime import datetime

# -------------------------------------------------------------------
# Global parameters and constants
# -------------------------------------------------------------------
date_time = datetime.today().strftime('%Y-%m-%d-%H:%M:%S') # Date and time to save output
MICE_DATASETS = 10
n_bootstraps = 1000
welch_range = 21
SEED = 42
rng = np.random.default_rng(seed=SEED)
NAME = 'name_to_save_output'

# Mapping of internal variable names to human-readable names for plots
change_names = {'clinical_frailty_scale': 'CFS', 'leukocytes': 'Leukocytes', 'albumin': 'Albumin', 'ngal': 'NGAL',
                'ddimer': 'D-dimer', 'il6': 'IL-6', 'age_at_admission': 'Age', 'meds_per_pat': '#Meds/Patient',
                'time_at_hospital_before_admission': 'Pre-ICU LoS',
                'no_CPR': 'DNR order',
                'thrombocytes': 'Thrombocytes', 'neutrofila': 'Neutrophils', 'A02BC': 'PPI medication',
                'cystatin_c': 'Cystatin C',
                'smoker': 'Smoker',
                'congestive_heart_failure': 'Congestive heart failure',
                'cancer': 'Cancer',
                'hypertension': 'Hypertension',
                'myocardial_infarction': 'Myocardial infarction',
                'peripheral_vascular_disease': 'Peripheral vascular disease',
                'cerebrovascular_insult': 'Cerebrovascular insult',
                'chronic_pulmonary_disease': 'Chronic pulmonary disease',
                'rheumatic_disease': 'Rheumatic disease',
                'peptic_ulcer_disease': 'Peptic ulcer disease',
                'diabetes_mellitus_uncompl': 'Diabetes mellitus (uncompl.)',
                'diabetes_mellitus_compl': 'Diabetes mellitus (compl.)',
                'chronic_kidney_disease': 'Chronic kidney disease',
                'imv_within_24h': 'IMV within 24h from ICU admission',
                'noradrenaline': 'Noradrenaline',
                'sex_male': 'Sex',
                'GCS': 'Glasgow Coma Scale',
                'cardiovascular_map': 'Cardiovascular mean arterial pressure',
                'creatinine': 'Creatinine',
                'bilirubin': 'Bilirubin',
                'ferritin': 'Ferritin',
                'laktat': 'Lactate',
                'pct': 'Procalcitonin',
                'pao2': 'Pa0$_2$',
                'paco2': 'PaC0$_2$',
                'aB_pH': 'pH (arterial blood)',
                'lymphocytes': 'Lymphocytes',
                'body_temperature': 'Body temperature (\N{DEGREE SIGN}C)',
                'heart_frequency': 'Heart rate (bpm)',
                'systolic_blood_pressure': 'Systolic blood pressure (mmHg)',
                'arterial_oxygen_tension': 'Arterial oxygen tension (mmHg)',
                'symptomatic_days': '#Symptomatic days',
                'habitual_creatinine': 'Habitual creatinine',
                'ICU_strain': 'ICU burden',
                'endostatin': 'Endostatin',
                'vcam': 'VCAM',
                'icam': 'ICAM',
                'calprotectin': 'Calprotectin',
                'pf_quotient': 'Pa0$_2$/FiO$_2$ Ratio',
                'survival_days_from_icu_start': '#Survival days from ICU admission',
                'mort': 'Dead within one year',
                'study_month': 'Study month'
                }

# -------------------------------------------------------------------
# Utility Function to rename variables
# -------------------------------------------------------------------
def plot_names(variables_to_rename, feat_name, feat_names):
    for old, new in variables_to_rename.items():
        if feat_name == old:
            feat_name = new
        elif old in feat_names:
            feat_names = [new if x == old else x for x in feat_names]
    return feat_name, feat_names

# -------------------------------------------------------------------
# SHAP label dictionary for consistent plotting and naming
# -------------------------------------------------------------------
labels = {
    'MAIN_EFFECT': "SHAP main effect value for\n%s",
    'INTERACTION_VALUE': "SHAP interaction value",
    'INTERACTION_EFFECT': "SHAP interaction value for\n%s and %s",
    'VALUE': "SHAP value",
    'GLOBAL_VALUE': "mean(|SHAP value|) (average impact on model output magnitude)",
    'VALUE_FOR': "SHAP value for\n%s",
    'PLOT_FOR': "SHAP plot for %s",
    'FEATURE': "Feature %s",
    'FEATURE_VALUE': "Feature value",
    'FEATURE_VALUE_LOW': "Low",
    'FEATURE_VALUE_HIGH': "High",
    'JOINT_VALUE': "Joint SHAP value",
    'MODEL_OUTPUT': "Model output value"
}


# -------------------------------------------------------------------
# Customization of the SHAP beeswarm plot to use on SHAP values collected in a training and internal validation loop
# -------------------------------------------------------------------
def beeswarm(
        shapvalues,  #  (n, 13)
        featurevalues: np.ndarray | None = None,  # (n, 13)
        featurenames: list | None = None,  #  (list of length 13)
        shap_values_shape: tuple | None = None,  # (n, 13)
        max_display: int | None = 14,
        order=Explanation.abs.mean(0),
        cluster_threshold=0.5,
        partition_tree=None,
        color=None,
        axis_color="#333333",
        alpha: float = 1.0,
        ax: plt.Axes | None = None,
        show: bool = True,
        log_scale: bool = False,
        color_bar: bool = True,
        s: float = 16,
        plot_size: Literal["auto"] | float | tuple[float, float] | None = "auto",
        color_bar_label: str = labels["FEATURE_VALUE"],
        group_remaining_features: bool = True,
):

    # Change the variable names for the plot
    feat_name = 'random_str_since_it_is_not_applicable'
    feat_names = featurenames
    _, feat_names = plot_names(change_names, feat_name, feat_names)

    values = shapvalues
    features = featurevalues
    if scipy.sparse.issparse(features):
        features = features.toarray()
    feature_names = feat_names  # rephrased names

    order = convert_ordering(order, values)

    # default color:
    if color is None:
        if features is not None:
            color = colors.red_blue
        else:
            color = colors.blue_rgb
    color = convert_color(color)

    idx2cat = None
    # convert from a DataFrame or other types
    if isinstance(features, pd.DataFrame):
        if feature_names is None:
            feature_names = features.columns
        # feature index to category flag
        idx2cat = features.dtypes.astype(str).isin(["object", "category"]).tolist()
        features = features.values
    elif isinstance(features, list):
        if feature_names is None:
            feature_names = features
        features = None
    elif (features is not None) and len(features.shape) == 1 and feature_names is None:
        feature_names = features
        features = None

    num_features = values.shape[1]

    if features is not None:
        shape_msg = "The shape of the shap_values matrix does not match the shape of the provided data matrix."
        if num_features - 1 == features.shape[1]:
            shape_msg += (
                " Constant offset? Pass shap_values[:,:-1]."
            )
            raise DimensionError(shape_msg)
        if num_features != features.shape[1]:
            raise DimensionError(shape_msg)

    if feature_names is None:
        feature_names = np.array([labels["FEATURE"] % str(i) for i in range(num_features)])

    if ax is None:
        ax = plt.gca()
    fig = ax.get_figure()
    assert isinstance(fig, Figure)  # type narrowing for mypy

    if log_scale:
        ax.set_xscale("symlog")

    # determine how many top features we will plot
    if max_display is None:
        max_display = len(feature_names)
    num_features = min(max_display, len(feature_names))

    # iteratively merge nodes until we can cut off the smallest feature values to stay within
    # num_features without breaking a cluster tree
    orig_inds = [[i] for i in range(len(feature_names))]
    orig_values = values.copy()  # SHAP values
    while True:
        feature_order = convert_ordering(order,
                                         Explanation(np.abs(values)))
        if partition_tree is not None:
            # compute the leaf order if we were to show (and so have the ordering respect) the whole partition tree
            clust_order = sort_inds(partition_tree, np.abs(values))

            # now relax the requirement to match the partition tree ordering for connections above cluster_threshold
            dist = scipy.spatial.distance.squareform(scipy.cluster.hierarchy.cophenet(partition_tree))
            feature_order = get_sort_order(dist, clust_order, cluster_threshold, feature_order)

            # if the last feature we can display is connected in a tree the next feature then we can't just cut
            # off the feature ordering, so we need to merge some tree nodes and then try again.
            if (
                    max_display < len(feature_order)
                    and dist[feature_order[max_display - 1], feature_order[max_display - 2]] <= cluster_threshold
            ):
                partition_tree, ind1, ind2 = merge_nodes(np.abs(values), partition_tree)
                for _ in range(len(values)):
                    values[:, ind1] += values[:, ind2]
                    values = np.delete(values, ind2, 1)
                    orig_inds[ind1] += orig_inds[ind2]
                    del orig_inds[ind2]
            else:
                break
        else:
            break

    # Build feature names, accounting for the fact that some features might be merged together
    feature_inds = feature_order[:max_display]
    feature_names_new = []
    for inds in orig_inds:
        if len(inds) == 1:
            feature_names_new.append(feat_names[inds[0]])
        elif len(inds) <= 2:
            feature_names_new.append(" + ".join([feat_names[i] for i in
                                                 inds]))
        else:
            max_ind = np.argmax(np.abs(orig_values).mean(0)[inds])
            feature_names_new.append(
                f"{feat_names[inds[max_ind]]} + {len(inds) - 1} other features")
    feature_names = feature_names_new

    # see how many individual (vs. grouped at the end) features are plotted
    include_grouped_remaining = num_features < len(values[0]) and group_remaining_features
    if include_grouped_remaining:
        num_cut = np.sum([len(orig_inds[feature_order[i]]) for i in range(num_features - 1, len(values[0]))])
        values[:, feature_order[num_features - 1]] = np.sum(
            [values[:, feature_order[i]] for i in range(num_features - 1, len(values[0]))], 0
        )

    # build y-tick labels
    yticklabels = [str(feature_names[i]) + ' (' + str(np.round(np.mean(np.abs(values[:, i])), 2)) + ')' for i in feature_inds]  # Adding the mean absolute SHAP value for each feature
    if include_grouped_remaining:
        yticklabels[-1] = f"Sum of {num_cut} other features"

    row_height = 0.4
    if plot_size == "auto":
        fig.set_size_inches(8, min(len(feature_order), max_display) * row_height + 1.5)
    elif isinstance(plot_size, (list, tuple)):
        fig.set_size_inches(plot_size[0], plot_size[1])
    elif plot_size is not None:
        fig.set_size_inches(8, min(len(feature_order), max_display) * plot_size + 1.5)
    ax.axvline(x=0, color="#999999", zorder=-1)

    # make beeswarm dots
    for pos, i in enumerate(reversed(feature_inds)):
        ax.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
        shaps = values[:, i]  # Shap values
        fvalues = None if features is None else features[:, i]  # original values
        f_inds = np.arange(len(shaps))
        np.random.shuffle(f_inds)
        if fvalues is not None:
            fvalues = fvalues[f_inds]
        shaps = shaps[f_inds]
        colored_feature = True
        try:
            if idx2cat is not None and idx2cat[i]:  # check categorical feature
                colored_feature = False
            else:
                fvalues = np.array(fvalues, dtype=np.float64)  # make sure this can be numeric
        except Exception:
            colored_feature = False
        N = len(shaps)
        nbins = 100
        quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
        inds_ = np.argsort(quant + np.random.randn(N) * 1e-6)
        layer = 0
        last_bin = -1
        ys = np.zeros(N)
        for ind in inds_:
            if quant[ind] != last_bin:
                layer = 0
            ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
            layer += 1
            last_bin = quant[ind]
        ys *= 0.9 * (row_height / np.max(ys + 1))

        if safe_isinstance(color, "matplotlib.colors.Colormap") and fvalues is not None and colored_feature is True:
            # trim the color range, but prevent the color range from collapsing
            vmin = np.nanpercentile(fvalues, 5)
            vmax = np.nanpercentile(fvalues, 95)
            if vmin == vmax:
                vmin = np.nanpercentile(fvalues, 1)
                vmax = np.nanpercentile(fvalues, 99)
                if vmin == vmax:
                    vmin = np.min(fvalues)
                    vmax = np.max(fvalues)
            if vmin > vmax:  # fixes rare numerical precision issues
                vmin = vmax

            if features is not None and features.shape[0] != len(shaps):
                emsg = "Feature and SHAP matrices must have the same number of rows."
                raise DimensionError(emsg)

            # plot the nan fvalues in the interaction feature as grey
            nan_mask = np.isnan(fvalues)
            ax.scatter(
                shaps[nan_mask],
                pos + ys[nan_mask],
                color="#777777",
                s=s,
                alpha=alpha,
                linewidth=0,
                zorder=3,
                rasterized=len(shaps) > 500,
            )

            # plot the non-nan fvalues colored by the trimmed feature value
            cvals = fvalues[np.invert(nan_mask)].astype(np.float64)
            cvals_imp = cvals.copy()
            cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
            cvals[cvals_imp > vmax] = vmax
            cvals[cvals_imp < vmin] = vmin
            ax.scatter(
                shaps[np.invert(nan_mask)],
                pos + ys[np.invert(nan_mask)],
                cmap=color,
                vmin=vmin,
                vmax=vmax,
                s=s,
                c=cvals,
                alpha=alpha,
                linewidth=0,
                zorder=3,
                rasterized=len(shaps) > 500,
            )
        else:
            if safe_isinstance(color, "matplotlib.colors.Colormap") and hasattr(color, "colors"):
                color = color.colors
            ax.scatter(
                shaps,
                pos + ys,
                s=s,
                alpha=alpha,
                linewidth=0,
                zorder=3,
                color=color if colored_feature else "#777777",
                rasterized=len(shaps) > 500,
            )

    # draw the color bar
    if safe_isinstance(color, "matplotlib.colors.Colormap") and color_bar and features is not None:
        import matplotlib.cm as cm

        m = cm.ScalarMappable(cmap=color)
        m.set_array([0, 1])
        cb = fig.colorbar(m, ax=ax, ticks=[0, 1], aspect=80)
        cb.set_ticklabels([labels["FEATURE_VALUE_LOW"], labels["FEATURE_VALUE_HIGH"]])
        cb.set_label(color_bar_label, size=12, labelpad=0)
        cb.ax.tick_params(labelsize=11, length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)  # type: ignore

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("none")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(color=axis_color, labelcolor=axis_color)
    ax.set_yticks(range(len(feature_inds)), list(reversed(yticklabels)), fontsize=13)
    ax.tick_params("y", length=20, width=0.5, which="major")
    ax.tick_params("x", labelsize=11)
    ax.set_ylim(-1, len(feature_inds))
    ax.set_xlabel(labels["VALUE"], fontsize=13)
    if show:
        plt.show()
    else:
        return ax

# -------------------------------------------------------------------
# Scatter plots with a LOESS curve for continuous variables
# -------------------------------------------------------------------
def plot_shap_feature(
        shap_values: Any,
        feature_values: pd.DataFrame,
        feature_name: int | str,
        feature_names: list,
        n_plots: int = 1
):

    # Change the variable names for the plot
    feat_name = feature_name
    feat_names = feature_names
    feat_name, feat_names = plot_names(change_names, feat_name, feat_names)

    # Color dots given another feature's values
    for i in range(n_plots):
        coloring_feature = feature_names[i]
        col_data = feature_values[coloring_feature]
        print('coloring feature', coloring_feature)

        # No added LOESS curve for a binary variable
        if feature_values[feature_name].isin([0, 1]).all():
            plt.figure()
            plt.axhline(y=0, color='k', linestyle='dashed')
            sc = plt.scatter(feature_values[feature_name], shap_values[feat_name],
                             cmap=colormaps['coolwarm'], c=col_data, s=40, label=f"SHAP value for {feat_name}")
            # Labels and colorbar
            plt.xlabel(feat_name)
            plt.ylabel("SHAP value")
            plt.title(
                "Impact on one-year mortality risk",
            )
            cbar = plt.colorbar(sc)
            cbar.set_label(feat_names[i])
            plt.grid(True)
            plt.tight_layout()
            save_as_svg(date_time, 'survival_dep_' + str(feature_name) +
                        '_' + str(coloring_feature), subfolder='dependency_plots')

            continue  # plot a dependency plot contra the next feature

        ### For all continuous variables
        raw_feature = np.asarray(feature_values[feature_name]).ravel() # Convert to NumPy arrays
        raw_shap = np.asarray(shap_values[feat_name]).ravel() # Convert to NumPy arrays

        # Grid for smoothing
        x_grid = np.linspace(np.min(raw_feature), np.max(raw_feature), len(raw_feature))
        x_grid = np.asarray(x_grid).ravel()

        # Apply loess
        xout, yout, wout = loess_1d.loess_1d(raw_feature, raw_shap,
                                             x_grid, degree=2, frac=0.6)

        # Bootstrap LOESS to estimate confidence interval
        boot_loess = []
        all_indices = feature_values.index.to_numpy()

        for ii in range(n_bootstraps):
            indices = rng.choice(
                all_indices,
                size=len(all_indices),
                replace=True
            )

            x_boot = feature_values.loc[indices, feature_name]
            y_boot = shap_values.loc[indices, feat_name]

            xout_boot, yout_boot, wout_boot = loess_1d.loess_1d(
                np.asarray(x_boot),
                np.asarray(y_boot),
                x_grid,
                degree=2,
                frac=0.6
            )

            boot_loess.append(yout_boot)

        boot_loess = np.array(boot_loess)
        ci_lower = np.percentile(boot_loess, 2.5, axis=0)
        ci_upper = np.percentile(boot_loess, 97.5, axis=0)


        # Scatter plots with LOESS
        fig, ax = plt.subplots()
        ax.axhline(y=0, color='k', linestyle='dashed')
        sc = plt.scatter(feature_values[feature_name], shap_values[feat_name],
                         cmap=colormaps['coolwarm'], c=col_data, s=40, label=f"SHAP value for {feat_name}")
        ax.plot(x_grid, yout, color='k', label=f"LOESS for SHAP for {feat_name}")
        ax.fill_between(x_grid, ci_lower, ci_upper, color='k', alpha=0.08)
        plt.xlabel(feat_name)

        # Mark reference points for polypharmacy comparison
        if feat_name == '#Meds/Patient':

            ax = plt.gca()

            for x_val in [2, 5]:
                mask = feature_values[feature_name] == x_val
                y_vals = shap_values.loc[mask, feat_name]

                if len(y_vals) > 0:
                    y_min, y_max = y_vals.min(), y_vals.max()

                    ellipse = Ellipse(
                        xy=(x_val, (y_min + y_max) / 2),
                        width=1.0,
                        height=(y_max - y_min) * 1.4,
                        fill=False,
                        edgecolor="red",
                        linewidth=2
                    )

                    ax.add_patch(ellipse)

        plt.ylabel("SHAP value")
        #plt.title("Variable effect on SHAP value for one-year mortality risk")
        cbar = plt.colorbar(sc)
        cbar.set_label(feat_names[i])
        ax.grid(True)
        plt.tight_layout()
        save_as_svg(date_time, 'survival_dep_' + str(feature_name) +
                    '_' + str(coloring_feature), subfolder='dependency_plots')

    return


# -------------------------------------------------------------------
# Calculate SHAP hazard ratios with 95% confidence intervals
# -------------------------------------------------------------------
def shap_hr(shap_mean, feature_mean, topfeatures, just_get_df = False):

    # Dataframe to store HR values (references values chosen using clinical expertise)
    hr_ci_dataframe = pd.DataFrame([], columns=topfeatures,
                                   index=['Ref', 'Step', 'HR_ref', 'CI_ref_low', 'CI_ref_high', 'HR_step',
                                          'CI_step_low', 'CI_step_high'])
    for name in topfeatures:
        if name == 'age_at_admission':
            hr_ci_dataframe.loc['Ref', name] = 60
            hr_ci_dataframe.loc['Step', name] = 75
        if name == 'cystatin_c':
            hr_ci_dataframe.loc['Ref', name] = 1
            hr_ci_dataframe.loc['Step', name] = 2.5
        if name == 'meds_per_pat':
            hr_ci_dataframe.loc['Ref', name] = 2
            hr_ci_dataframe.loc['Step', name] = 5
        if name == 'il6':
            hr_ci_dataframe.loc['Ref', name] = 66
            hr_ci_dataframe.loc['Step', name] = 286.5
        if name == 'time_at_hospital_before_admission':
            hr_ci_dataframe.loc['Ref', name] = 0
            hr_ci_dataframe.loc['Step', name] = 7
        if name == 'ICU_strain':
            hr_ci_dataframe.loc['Ref', name] = 20
            hr_ci_dataframe.loc['Step', name] = 45
        if name == 'calprotectin':
            hr_ci_dataframe.loc['Ref', name] = 5
            hr_ci_dataframe.loc['Step', name] = 10
        if name == 'laktat':
            hr_ci_dataframe.loc['Ref', name] = 2
            hr_ci_dataframe.loc['Step', name] = 4
        if name == 'BMI':
            hr_ci_dataframe.loc['Ref', name] = 25
            hr_ci_dataframe.loc['Step', name] = 35
        if name == 'Thrombocytes':
            hr_ci_dataframe.loc['Ref', name] = 150
            hr_ci_dataframe.loc['Step', name] = 300
        if name == 'ferritin':
            hr_ci_dataframe.loc['Ref', name] = 712.5
            hr_ci_dataframe.loc['Step', name] = 2255
        if name == 'systolic_blood_pressure':
            hr_ci_dataframe.loc['Ref', name] = 100
            hr_ci_dataframe.loc['Step', name] = 120
        if name == 'lymphocytes':
            hr_ci_dataframe.loc['Ref', name] = 0.4
            hr_ci_dataframe.loc['Step', name] = 0.8


    # Just return the dataframe to use in other function
    if just_get_df:
        return hr_ci_dataframe

    # Calculate HR and CI
    else:

        ### HR SHAP for binary variables ###
        # Get binary variables
        bin_var = ['C03CA', 'A10BK', 'N06AX', 'N05BA', 'N03AX', 'B01AF', 'C01DA', 'N06AB', 'B03BA', 'N02AA', 'N05CF',
                   'R03AC', 'R03AK', 'H02AB', 'A10BJ', 'C10AA', 'B01AC', 'C07AB', 'A02BC', 'C08CA', 'A12AX', 'N02BE',
                   'B03BB', 'A10BA', 'Y92BA', 'Y92AD', 'H03AA', 'M04AA', 'congestive_heart_failure', 'cancer', 'NOAK',
                   'hypertension', 'myocardial_infarction', 'peripheral_vascular_disease', 'cerebrovascular_insult',
                   'chronic_pulmonary_disease', 'rheumatic_disease', 'peptic_ulcer_disease', 'diabetes_mellitus_uncompl',
                   'diabetes_mellitus_compl', 'chronic_kidney_disease', 'imv_within_24h', 'sex_male', 'no_CPR', 'ACEi',
                   'ARB']

        ########### Welch test for binary variables #####################
        shap_bin_stats = pd.DataFrame(columns=bin_var)
        for variable_bin in bin_var:
            zero_idx = feature_mean[variable_bin].loc[feature_mean[variable_bin] == 0].index
            one_idx = feature_mean[variable_bin].loc[feature_mean[variable_bin] == 1].index

            var0 = np.var(shap_mean[variable_bin].loc[
                              shap_mean[variable_bin].index.isin(zero_idx)], ddof=1)
            var1 = np.var(shap_mean[variable_bin].loc[
                              shap_mean[variable_bin].index.isin(one_idx)], ddof=1)

            n0 = len(shap_mean[variable_bin].loc[
                         shap_mean[variable_bin].index.isin(zero_idx)])
            n1 = len(shap_mean[variable_bin].loc[
                         shap_mean[variable_bin].index.isin(one_idx)])

            # Group mean
            mean0 = np.mean(shap_mean[variable_bin].loc[
                                shap_mean[variable_bin].index.isin(zero_idx)])
            mean1 = np.mean(shap_mean[variable_bin].loc[
                                shap_mean[variable_bin].index.isin(one_idx)])

            # Mean difference
            difference = mean1 - mean0

            # Welchs stnadard error
            se = np.sqrt(var1 / n1 + var0 / n0)

            # Welch degrees of freedom
            numerator = (var1 / n1 + var0 / n0) ** 2

            denominator = (
                    (var1 / n1) ** 2 / (n1 - 1)
                    + (var0 / n0) ** 2 / (n0 - 1)
            )

            df = numerator / denominator

            # 95% CI
            alpha = 0.05
            t_critical = t.ppf(1 - alpha / 2, df)

            ci_lower = difference - t_critical * se
            ci_upper = difference + t_critical * se

            resultat = st.ttest_ind(
                shap_mean[variable_bin].loc[
                    shap_mean[variable_bin].index.isin(one_idx)],
                shap_mean[variable_bin].loc[
                    shap_mean[variable_bin].index.isin(zero_idx)],
                equal_var=False,  # Welch
                alternative="two-sided"
            )

            print(f"Number in group 0: {n0}")
            print(f"Number in group 1: {n1}")
            print(f"Mean group 0: {mean0:.4f}")
            print(f"Mean group 1: {mean1:.4f}")
            print(f"Difference, group 1 - group 0: {difference:.4f}")
            print(f"Standard error: {se:.4f}")
            print(f"Welch degrees of freedom: {df:.2f}")
            print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
            print(f"HR 95% CI: [{np.exp(difference):.4f}, {np.exp(ci_lower):.4f}, {np.exp(ci_upper):.4f}]")

            print(variable_bin)
            print(f"Welchs t statistics: {resultat.statistic:.4f}")
            print(f"Twosided p value: {resultat.pvalue:.6f}")
            print(f"HR 95% CI: [{np.exp(difference):.2f}, {np.exp(ci_lower):.2f}, {np.exp(ci_upper):.2f}]")

            # Add information to df
            shap_bin_stats.loc['log_hr', variable_bin] = difference
            shap_bin_stats.loc['std', variable_bin] = se
            shap_bin_stats.loc['p-value', variable_bin] = resultat.pvalue
            shap_bin_stats.loc['hazard_ratio', variable_bin] = np.exp(difference)

            # The bootstrapped 95% CI
            shap_bin_stats.loc['ci_low_lg', variable_bin] = ci_lower
            shap_bin_stats.loc['ci_high_lg', variable_bin] = ci_upper
            shap_bin_stats.loc['ci_low', variable_bin] = np.exp(ci_lower)
            shap_bin_stats.loc['ci_high', variable_bin] = np.exp(ci_upper)
            ##########################################

            # Number variable_bin == 1
            shap_bin_stats.loc['n=1', variable_bin] = n1
            # Number variable_bin == 0
            shap_bin_stats.loc['n=0', variable_bin] = n0

        # Change the variable name of the dataframe
        shap_bin_stats.rename(columns=change_names, inplace=True)

        # Save df for binary variables
        excel(date_time, shap_bin_stats, str(NAME) + '_shap_bin_stats', subfolder='SHAP')

        ### HR SHAP for all ATC codes  ###
        # Get ATC variables
        atc_var = ['C03CA', 'A10BK', 'N06AX', 'N05BA', 'N03AX', 'B01AF', 'C01DA', 'N06AB', 'B03BA', 'N02AA', 'N05CF',
                   'R03AC', 'R03AK', 'H02AB', 'A10BJ', 'C10AA', 'B01AC', 'C07AB', 'A02BC', 'C08CA', 'A12AX', 'N02BE',
                   'B03BB', 'A10BA', 'Y92BA', 'Y92AD', 'H03AA', 'M04AA', 'ACEi', 'ARB']

        ########### Welch test for binary ATC variables #####################
        shap_atc_stats = pd.DataFrame(columns=atc_var)
        for variable_atc in atc_var:
            zero_idx = feature_mean[variable_atc].loc[feature_mean[variable_atc] == 0].index
            one_idx = feature_mean[variable_atc].loc[feature_mean[variable_atc] == 1].index

            var0 = np.var(shap_mean[variable_atc].loc[
                              shap_mean[variable_atc].index.isin(zero_idx)], ddof=1)
            var1 = np.var(shap_mean[variable_atc].loc[
                              shap_mean[variable_atc].index.isin(one_idx)], ddof=1)

            n0 = len(shap_mean[variable_atc].loc[
                         shap_mean[variable_atc].index.isin(zero_idx)])
            n1 = len(shap_mean[variable_atc].loc[
                         shap_mean[variable_atc].index.isin(one_idx)])

            # Group mean
            mean0 = np.mean(shap_mean[variable_atc].loc[
                                shap_mean[variable_atc].index.isin(zero_idx)])
            mean1 = np.mean(shap_mean[variable_atc].loc[
                                shap_mean[variable_atc].index.isin(one_idx)])

            # Mean difference
            difference = mean1 - mean0

            # Welchs stnadard error
            se = np.sqrt(var1 / n1 + var0 / n0)

            # Welch degrees of freedom
            numerator = (var1 / n1 + var0 / n0) ** 2

            denominator = (
                    (var1 / n1) ** 2 / (n1 - 1)
                    + (var0 / n0) ** 2 / (n0 - 1)
            )

            df = numerator / denominator

            # 95% CI
            alpha = 0.05
            t_critical = t.ppf(1 - alpha / 2, df)

            ci_lower = difference - t_critical * se
            ci_upper = difference + t_critical * se

            resultat = st.ttest_ind(
                shap_mean[variable_atc].loc[
                    shap_mean[variable_atc].index.isin(one_idx)],
                shap_mean[variable_atc].loc[
                    shap_mean[variable_atc].index.isin(zero_idx)],
                equal_var=False,  # Welch
                alternative="two-sided"
            )

            print(f"Number in group 0: {n0}")
            print(f"Number in group 1: {n1}")
            print(f"Mean group 0: {mean0:.4f}")
            print(f"Mean group 1: {mean1:.4f}")
            print(f"Difference, group 1 - group 0: {difference:.4f}")
            print(f"Standard error: {se:.4f}")
            print(f"Welch degrees of freedom: {df:.2f}")
            print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
            print(f"HR 95% CI: [{np.exp(difference):.4f}, {np.exp(ci_lower):.4f}, {np.exp(ci_upper):.4f}]")

            print(variable_atc)
            print(f"Welchs t statistics: {resultat.statistic:.4f}")
            print(f"Twosided p value: {resultat.pvalue:.6f}")
            print(f"HR 95% CI: [{np.exp(difference):.2f}, {np.exp(ci_lower):.2f}, {np.exp(ci_upper):.2f}]")

            # Add information to df
            shap_atc_stats.loc['log_hr', variable_atc] = difference
            shap_atc_stats.loc['std', variable_atc] = se
            shap_atc_stats.loc['p-value', variable_atc] = resultat.pvalue
            shap_atc_stats.loc['hazard_ratio', variable_atc] = np.exp(difference)

            # The bootstrapped 95% CI
            shap_atc_stats.loc['ci_low_lg', variable_atc] = ci_lower
            shap_atc_stats.loc['ci_high_lg', variable_atc] = ci_upper
            shap_atc_stats.loc['ci_low', variable_atc] = np.exp(ci_lower)
            shap_atc_stats.loc['ci_high', variable_atc] = np.exp(ci_upper)
            ##########################################


            # Number variable_atc == 1
            shap_atc_stats.loc['n=1', variable_atc] = n1
            # Number variable_atc == 0
            shap_atc_stats.loc['n=0', variable_atc] = n0

        # Change the variable name of the dataframe
        shap_atc_stats.rename(columns=change_names, inplace=True)

        # Save df for atc variables
        excel(date_time, shap_atc_stats, str(NAME) + '_shap_atc_stats', subfolder='SHAP')

        ### HR SHAP for ordinal variables ###
        smoker = ['smoker_01', 'smoker_12', 'smoker_02']
        noradre = ['noradrenaline_01', 'noradrenaline_12', 'noradrenaline_02']
        ccis = ['cci_01', 'cci_12', 'cci_23', 'cci_34', 'cci_45', 'cci_56', 'cci_67', 'cci_78', 'cci_89', 'cci_910',
                'cci_1011', 'cci_36']  # only full-case

        ########### Welch test for smoker #####################
        shap_ordinal_stats = pd.DataFrame(columns=smoker + noradre +ccis)
        for level, label in enumerate(smoker):
            if level == 0:
                lower_idx = feature_mean["smoker"].loc[feature_mean["smoker"] == 0].index
                higher_idx = feature_mean["smoker"].loc[feature_mean["smoker"] == 1].index

            if level == 1:
                lower_idx = feature_mean["smoker"].loc[feature_mean["smoker"] == 1].index
                higher_idx = feature_mean["smoker"].loc[feature_mean["smoker"] == 2].index

            if level == 2:
                lower_idx = feature_mean["smoker"].loc[feature_mean["smoker"] == 0].index
                higher_idx = feature_mean["smoker"].loc[feature_mean["smoker"] == 2].index

            var0 = np.var(shap_mean["smoker"].loc[
                              shap_mean["smoker"].index.isin(lower_idx)], ddof=1)
            var1 = np.var(shap_mean["smoker"].loc[
                              shap_mean["smoker"].index.isin(higher_idx)], ddof=1)

            n0 = len(shap_mean["smoker"].loc[
                         shap_mean["smoker"].index.isin(lower_idx)])
            n1 = len(shap_mean["smoker"].loc[
                         shap_mean["smoker"].index.isin(higher_idx)])

            # group mean
            mean0 = np.mean(shap_mean["smoker"].loc[
                                shap_mean["smoker"].index.isin(lower_idx)])
            mean1 = np.mean(shap_mean["smoker"].loc[
                                shap_mean["smoker"].index.isin(higher_idx)])

            # Difference group1 - group0
            difference = mean1 - mean0

            # Welchs standard error
            se = np.sqrt(var1 / n1 + var0 / n0)

            # Welch degrees of freedom
            numerator = (var1 / n1 + var0 / n0) ** 2

            denominator = (
                    (var1 / n1) ** 2 / (n1 - 1)
                    + (var0 / n0) ** 2 / (n0 - 1)
            )

            df = numerator / denominator

            # 95% CI
            alpha = 0.05
            t_critical = t.ppf(1 - alpha / 2, df)

            ci_lower = difference - t_critical * se
            ci_upper = difference + t_critical * se

            resultat = st.ttest_ind(
                shap_mean["smoker"].loc[
                    shap_mean["smoker"].index.isin(lower_idx)],
                shap_mean["smoker"].loc[
                    shap_mean["smoker"].index.isin(higher_idx)],
                equal_var=False,  # Welch
                alternative="two-sided"
            )

            print(f"Number in group 0: {n0}")
            print(f"Number in group 1: {n1}")
            print(f"Mean group 0: {mean0:.4f}")
            print(f"Mean group 1: {mean1:.4f}")
            print(f"Difference, group 1 - group 0: {difference:.4f}")
            print(f"Standard error: {se:.4f}")
            print(f"Welch degrees of freedom: {df:.2f}")
            print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
            print(f"HR 95% CI: [{np.exp(difference):.4f}, {np.exp(ci_lower):.4f}, {np.exp(ci_upper):.4f}]")

            print("smoker")
            print(f"Welchs t statistics: {resultat.statistic:.4f}")
            print(f"Twosided p value: {resultat.pvalue:.6f}")
            print(f"HR 95% CI: [{np.exp(difference):.2f}, {np.exp(ci_lower):.2f}, {np.exp(ci_upper):.2f}]")

            # Add information to df
            shap_ordinal_stats.loc['log_hr', label] = difference
            shap_ordinal_stats.loc['std', label] = se
            shap_ordinal_stats.loc['p-value', label] = resultat.pvalue
            shap_ordinal_stats.loc['hazard_ratio', label] = np.exp(difference)

            # The 95% CI
            shap_ordinal_stats.loc['ci_low_lg',label] = ci_lower
            shap_ordinal_stats.loc['ci_high_lg', label] = ci_upper
            shap_ordinal_stats.loc['ci_low', label] = np.exp(ci_lower)
            shap_ordinal_stats.loc['ci_high', label] = np.exp(ci_upper)
            ##########################################


            # Number label == higher
            shap_ordinal_stats.loc['n=higher', label] = n1
            # Number label == lower
            shap_ordinal_stats.loc['n=lower', label] = n0

        ########### Welch test for noradrenaline #####################

        for level, label in enumerate(noradre):
            if level == 0:
                lower_idx = feature_mean["noradrenaline"].loc[feature_mean["noradrenaline"] == 0].index
                higher_idx = feature_mean["noradrenaline"].loc[feature_mean["noradrenaline"] == 1].index

            if level == 1:
                lower_idx = feature_mean["noradrenaline"].loc[feature_mean["noradrenaline"] == 1].index
                higher_idx = feature_mean["noradrenaline"].loc[feature_mean["noradrenaline"] == 2].index

            if level == 2:
                lower_idx = feature_mean["noradrenaline"].loc[feature_mean["noradrenaline"] == 0].index
                higher_idx = feature_mean["noradrenaline"].loc[feature_mean["noradrenaline"] == 2].index

            var0 = np.var(shap_mean["noradrenaline"].loc[
                              shap_mean["noradrenaline"].index.isin(lower_idx)], ddof=1)
            var1 = np.var(shap_mean["noradrenaline"].loc[
                              shap_mean["noradrenaline"].index.isin(higher_idx)], ddof=1)

            n0 = len(shap_mean["noradrenaline"].loc[
                         shap_mean["noradrenaline"].index.isin(lower_idx)])
            n1 = len(shap_mean["noradrenaline"].loc[
                         shap_mean["noradrenaline"].index.isin(higher_idx)])

            # Group mean
            mean0 = np.mean(shap_mean["noradrenaline"].loc[
                                shap_mean["noradrenaline"].index.isin(lower_idx)])
            mean1 = np.mean(shap_mean["noradrenaline"].loc[
                                shap_mean["noradrenaline"].index.isin(higher_idx)])

            # Difference group 1 - group 2
            difference = mean1 - mean0

            # Welchs standard error
            se = np.sqrt(var1 / n1 + var0 / n0)

            # Welch degrees of freedom
            numerator = (var1 / n1 + var0 / n0) ** 2

            denominator = (
                    (var1 / n1) ** 2 / (n1 - 1)
                    + (var0 / n0) ** 2 / (n0 - 1)
            )

            df = numerator / denominator

            # 95% CI
            alpha = 0.05
            t_critical = t.ppf(1 - alpha / 2, df)

            ci_lower = difference - t_critical * se
            ci_upper = difference + t_critical * se

            resultat = st.ttest_ind(
                shap_mean["noradrenaline"].loc[
                    shap_mean["noradrenaline"].index.isin(lower_idx)],
                shap_mean["noradrenaline"].loc[
                    shap_mean["noradrenaline"].index.isin(higher_idx)],
                equal_var=False,  # Welch
                alternative="two-sided"
            )

            print("noradrenaline")
            print(f"Number in group 0: {n0}")
            print(f"Number in group 1: {n1}")
            print(f"Mean group 0: {mean0:.4f}")
            print(f"Mean group 1: {mean1:.4f}")
            print(f"Difference, group 1 - group 0: {difference:.4f}")
            print(f"Standard error: {se:.4f}")
            print(f"Welch degrees of freedom: {df:.2f}")
            print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
            print(f"HR 95% CI: [{np.exp(difference):.4f}, {np.exp(ci_lower):.4f}, {np.exp(ci_upper):.4f}]")
            print(f"Welchs t statistics: {resultat.statistic:.4f}")
            print(f"Twosided p value: {resultat.pvalue:.6f}")
            print(f"HR 95% CI: [{np.exp(difference):.2f}, {np.exp(ci_lower):.2f}, {np.exp(ci_upper):.2f}]")

            # Add information to df
            shap_ordinal_stats.loc['log_hr', label] = difference
            shap_ordinal_stats.loc['std', label] = se
            shap_ordinal_stats.loc['p-value', label] = resultat.pvalue
            shap_ordinal_stats.loc['hazard_ratio', label] = np.exp(difference)

            # The 95% CI
            shap_ordinal_stats.loc['ci_low_lg', label] = ci_lower
            shap_ordinal_stats.loc['ci_high_lg', label] = ci_upper
            shap_ordinal_stats.loc['ci_low', label] = np.exp(ci_lower)
            shap_ordinal_stats.loc['ci_high', label] = np.exp(ci_upper)

            # Number label == higher
            shap_ordinal_stats.loc['n=higher', label] = n1
            # Number label == lower
            shap_ordinal_stats.loc['n=lower', label] = n0

        ########### Welch test for CCI #####################

        for level, label in enumerate(ccis):
            if level == 0:
                lower_idx = feature_mean["CCI"].loc[feature_mean["CCI"] == 0].index
                higher_idx = feature_mean["CCI"].loc[feature_mean["CCI"] == 1].index

            if level == 1:
                lower_idx = feature_mean["CCI"].loc[feature_mean["CCI"] == 1].index
                higher_idx = feature_mean["CCI"].loc[feature_mean["CCI"] == 2].index

            if level == 2:
                lower_idx = feature_mean["CCI"].loc[feature_mean["CCI"] == 2].index
                higher_idx = feature_mean["CCI"].loc[feature_mean["CCI"] == 3].index

            if level == 3:
                lower_idx = feature_mean["CCI"].loc[feature_mean["CCI"] == 3].index
                higher_idx = feature_mean["CCI"].loc[feature_mean["CCI"] == 4].index

            if level == 4:
                lower_idx = feature_mean["CCI"].loc[feature_mean["CCI"] == 4].index
                higher_idx = feature_mean["CCI"].loc[feature_mean["CCI"] == 5].index

            if level == 5:
                lower_idx = feature_mean["CCI"].loc[feature_mean["CCI"] == 5].index
                higher_idx = feature_mean["CCI"].loc[feature_mean["CCI"] == 6].index

            if level == 6:
                lower_idx = feature_mean["CCI"].loc[feature_mean["CCI"] == 6].index
                higher_idx = feature_mean["CCI"].loc[feature_mean["CCI"] == 7].index

            if level == 7:
                lower_idx = feature_mean["CCI"].loc[feature_mean["CCI"] == 7].index
                higher_idx = feature_mean["CCI"].loc[feature_mean["CCI"] == 8].index

            if level == 8:
                lower_idx = feature_mean["CCI"].loc[feature_mean["CCI"] == 8].index
                higher_idx = feature_mean["CCI"].loc[feature_mean["CCI"] == 9].index

            if level == 9:
                lower_idx = feature_mean["CCI"].loc[feature_mean["CCI"] == 9].index
                higher_idx = feature_mean["CCI"].loc[feature_mean["CCI"] == 10].index

            if level == 10:
                lower_idx = feature_mean["CCI"].loc[feature_mean["CCI"] == 10].index
                higher_idx = feature_mean["CCI"].loc[feature_mean["CCI"] == 11].index

            if level == 11:
                lower_idx = feature_mean["CCI"].loc[feature_mean["CCI"] == 3].index
                higher_idx = feature_mean["CCI"].loc[feature_mean["CCI"] == 6].index

            var0 = np.var(shap_mean["CCI"].loc[
                              shap_mean["CCI"].index.isin(lower_idx)], ddof=1)
            var1 = np.var(shap_mean["CCI"].loc[
                              shap_mean["CCI"].index.isin(higher_idx)], ddof=1)

            n0 = len(shap_mean["CCI"].loc[
                         shap_mean["CCI"].index.isin(lower_idx)])
            n1 = len(shap_mean["CCI"].loc[
                         shap_mean["CCI"].index.isin(higher_idx)])

            # If there are not enough records in a group, continue the loop to the next group
            if n0 == 0 or n1 == 0:
                # Number label == higher
                shap_ordinal_stats.loc['n=higher', label] = n1
                # Number label == lower
                shap_ordinal_stats.loc['n=lower', label] = n0
                continue

            # Group mean
            mean0 = np.mean(shap_mean["CCI"].loc[
                                shap_mean["CCI"].index.isin(lower_idx)])
            mean1 = np.mean(shap_mean["CCI"].loc[
                                shap_mean["CCI"].index.isin(higher_idx)])

            # Difference in group 1 - group 0
            difference = mean1 - mean0

            # Welchs standard error
            se = np.sqrt(var1 / n1 + var0 / n0)

            # Welch degrees of freedom
            numerator = (var1 / n1 + var0 / n0) ** 2

            denominator = (
                    (var1 / n1) ** 2 / (n1 - 1)
                    + (var0 / n0) ** 2 / (n0 - 1)
            )

            df = numerator / denominator

            # 95% CI
            alpha = 0.05
            t_critical = t.ppf(1 - alpha / 2, df)

            ci_lower = difference - t_critical * se
            ci_upper = difference + t_critical * se

            resultat = st.ttest_ind(
                shap_mean["CCI"].loc[
                    shap_mean["CCI"].index.isin(lower_idx)],
                shap_mean["CCI"].loc[
                    shap_mean["CCI"].index.isin(higher_idx)],
                equal_var=False,  # Welch
                alternative="two-sided"
            )

            print("CCI")
            print(f"Number in group 0: {n0}")
            print(f"Number in group 1: {n1}")
            print(f"Mean group 0: {mean0:.4f}")
            print(f"Mean group 1: {mean1:.4f}")
            print(f"Difference, group 1 - group 0: {difference:.4f}")
            print(f"Standard error: {se:.4f}")
            print(f"Welch degrees of freedom: {df:.2f}")
            print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
            print(f"HR 95% CI: [{np.exp(difference):.4f}, {np.exp(ci_lower):.4f}, {np.exp(ci_upper):.4f}]")
            print(f"Welchs t statistics: {resultat.statistic:.4f}")
            print(f"Twosided p value: {resultat.pvalue:.6f}")
            print(f"HR 95% CI: [{np.exp(difference):.2f}, {np.exp(ci_lower):.2f}, {np.exp(ci_upper):.2f}]")

            # Add information to df
            shap_ordinal_stats.loc['log_hr', label] = difference
            shap_ordinal_stats.loc['std', label] = se
            shap_ordinal_stats.loc['p-value', label] = resultat.pvalue
            shap_ordinal_stats.loc['hazard_ratio', label] = np.exp(difference)

            # The 95% CI
            shap_ordinal_stats.loc['ci_low_lg', label] = ci_lower
            shap_ordinal_stats.loc['ci_high_lg', label] = ci_upper
            shap_ordinal_stats.loc['ci_low', label] = np.exp(ci_lower)
            shap_ordinal_stats.loc['ci_high', label] = np.exp(ci_upper)


            # Number label == higher
            shap_ordinal_stats.loc['n=higher', label] = n1
            # Number label == lower
            shap_ordinal_stats.loc['n=lower', label] = n0


        # Change the variable name of the dataframe
        shap_ordinal_stats.rename(columns=change_names, inplace=True)

        # Save df for binary variables
        excel(date_time, shap_ordinal_stats, str(NAME) + '_shap_ordinal_stats', subfolder='SHAP')

        ### HR SHAP for continuous variables ###
        ######## REMADE AS WELCH WITH AN INTERVAL FOR THE REF AND STEP POINT ###########
        def shap_stats_cont(variables, features, shaps, hr_ci_df):

            shap_cont_stats = pd.DataFrame(columns=variables)
            discret_values = ['age_at_admission', 'time_at_hospital_before_admission', 'meds_per_pat'] # excluding ICU burden since there are so few records per value
            continuous_values =  [var for var in variables if var not in discret_values]

            ##############################################################
            for cont_var in hr_ci_df.columns.to_list():
                if cont_var in discret_values:

                    # Find indices of reference and step records
                    ref_idx = features.loc[features[cont_var] == hr_ci_df.loc["Ref", cont_var]].index
                    step_idx = features.loc[features[cont_var] == hr_ci_df.loc["Step", cont_var]].index

                    var0 = np.var(shaps[cont_var].loc[
                                      shaps[cont_var].index.isin(ref_idx)], ddof=1)
                    var1 = np.var(shaps[cont_var].loc[
                                      shaps[cont_var].index.isin(step_idx)], ddof=1)

                    n0 = len(shaps[cont_var].loc[
                                 shaps[cont_var].index.isin(ref_idx)])
                    n1 = len(shaps[cont_var].loc[
                                 shaps[cont_var].index.isin(step_idx)])

                    # If there are not enough records in a group, continue the loop to the next group
                    if n0 == 0 or n1 == 0:
                        # Number label == higher
                        shap_ordinal_stats.loc['n=step', cont_var] = n1
                        # Number label == lower
                        shap_ordinal_stats.loc['n=ref', cont_var] = n0
                        continue

                    # Group mean
                    mean0 = np.mean(shaps[cont_var].loc[
                                        shaps[cont_var].index.isin(ref_idx)])
                    mean1 = np.mean(shaps[cont_var].loc[
                                        shaps[cont_var].index.isin(step_idx)])

                    # Difference group 1 - group 0
                    difference = mean1 - mean0

                    # Welch standard error
                    se = np.sqrt(var1 / n1 + var0 / n0)

                    # Welch degrees of freedom
                    numerator = (var1 / n1 + var0 / n0) ** 2

                    denominator = (
                            (var1 / n1) ** 2 / (n1 - 1)
                            + (var0 / n0) ** 2 / (n0 - 1)
                    )

                    df = numerator / denominator

                    # 95% CI
                    alpha = 0.05
                    t_critical = t.ppf(1 - alpha / 2, df)

                    ci_lower = difference - t_critical * se
                    ci_upper = difference + t_critical * se

                    resultat = st.ttest_ind(
                        shaps[cont_var].loc[
                            shaps[cont_var].index.isin(ref_idx)],
                        shaps[cont_var].loc[
                            shaps[cont_var].index.isin(step_idx)],
                        equal_var=False,  # Welch
                        alternative="two-sided"
                    )

                    print(cont_var)
                    print(f"Number in group 0: {n0}")
                    print(f"Number in group 1: {n1}")
                    print(f"Mean group 0: {mean0:.4f}")
                    print(f"Mean group 1: {mean1:.4f}")
                    print(f"Difference, group 1 - group 0: {difference:.4f}")
                    print(f"Standard error: {se:.4f}")
                    print(f"Welch degrees of freedom: {df:.2f}")
                    print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
                    print(f"HR 95% CI: [{np.exp(difference):.4f}, {np.exp(ci_lower):.4f}, {np.exp(ci_upper):.4f}]")
                    print(f"Welchs t statistics: {resultat.statistic:.4f}")
                    print(f"Twosided p value: {resultat.pvalue:.6f}")
                    print(f"HR 95% CI: [{np.exp(difference):.2f}, {np.exp(ci_lower):.2f}, {np.exp(ci_upper):.2f}]")

                    # Add information to df
                    shap_cont_stats.loc['log_hr', cont_var] = difference
                    shap_cont_stats.loc['std', cont_var] = se
                    shap_cont_stats.loc['p-value', cont_var] = resultat.pvalue
                    shap_cont_stats.loc['hazard_ratio', cont_var] = np.exp(difference)

                    # The 95% CI
                    shap_cont_stats.loc['ci_low_lg', cont_var] = ci_lower
                    shap_cont_stats.loc['ci_high_lg', cont_var] = ci_upper
                    shap_cont_stats.loc['ci_low', cont_var] = np.exp(ci_lower)
                    shap_cont_stats.loc['ci_high', cont_var] = np.exp(ci_upper)

                    # Number label == higher
                    shap_cont_stats.loc['n=higher', cont_var] = n1
                    # Number label == lower
                    shap_cont_stats.loc['n=lower', cont_var] = n0

                elif cont_var in continuous_values:

                    if np.isnan(hr_ci_df.loc["Ref", cont_var]) or np.isnan(hr_ci_df.loc["Step", cont_var]):
                        print(cont_var,"has ref",hr_ci_df.loc["Ref", cont_var],"and step",hr_ci_df.loc["Step", cont_var])
                        continue

                    # Find indices of the n values closest to Ref/Step
                    ref_idc = np.argsort(np.abs(np.array(features[cont_var]) - hr_ci_df.loc["Ref", cont_var]))[:welch_range] # argsort returns the indices that would sort an array
                    step_idc = np.argsort(np.abs(np.array(features[cont_var]) - hr_ci_df.loc["Step", cont_var]))[:welch_range]

                    text = "\n\n" + str(cont_var) + ": " + str(hr_ci_df.loc["Ref", cont_var]) + ", " +  str(hr_ci_df.loc["Step", cont_var]) + "\n" + str(features.loc[ref_idc, cont_var].min()) + ", " +  str(features.loc[ref_idc, cont_var].max()) + "\n" + str(features.loc[step_idc, cont_var].min()) + ", " +  str(features.loc[step_idc, cont_var].max())
                    write_to_file(date_time, "welch_ranges", text, read_text=True)

                    # Find the lowest and highest mean feature value for each reference point
                    print("\n", cont_var, hr_ci_df.loc["Ref", cont_var], hr_ci_df.loc["Step", cont_var])
                    print(features.loc[ref_idc, cont_var].min(), features.loc[ref_idc, cont_var].max())
                    print(features.loc[step_idc, cont_var].min(), features.loc[step_idc, cont_var].max())


                    # Count how many selected values are below, above, or equal to ref_value
                    n_below = sum(v < hr_ci_df.loc["Ref", cont_var] for v in features[cont_var].loc[features[cont_var].index.isin(ref_idc)].to_list())
                    n_above = sum(v > hr_ci_df.loc["Ref", cont_var] for v in features[cont_var].loc[features[cont_var].index.isin(ref_idc)].to_list())
                    n_equal = sum(v == hr_ci_df.loc["Ref", cont_var] for v in features[cont_var].loc[features[cont_var].index.isin(ref_idc)].to_list())

                    print(cont_var, hr_ci_df.loc["Ref", cont_var], hr_ci_df.loc["Step", cont_var])

                    print(
                        f"Of the {welch_range} closest values to Ref, "
                        f"{n_below} are below the reference value, "
                        f"{n_above} are above it, and "
                        f"{n_equal} are equal to it."
                    )

                    n_below = sum(v < hr_ci_df.loc["Step", cont_var] for v in features[cont_var].loc[features[cont_var].index.isin(step_idc)].to_list())
                    n_above = sum(v > hr_ci_df.loc["Step", cont_var] for v in features[cont_var].loc[features[cont_var].index.isin(step_idc)].to_list())
                    n_equal = sum(v == hr_ci_df.loc["Step", cont_var] for v in features[cont_var].loc[features[cont_var].index.isin(step_idc)].to_list())

                    print(
                        f"Of the {welch_range} closest values to Step, "
                        f"{n_below} are below the reference value, "
                        f"{n_above} are above it, and "
                        f"{n_equal} are equal to it."
                    )

                    var0 = np.var(shaps[cont_var].loc[
                                      shaps[cont_var].index.isin(ref_idc)], ddof=1)
                    var1 = np.var(shaps[cont_var].loc[
                                      shaps[cont_var].index.isin(step_idc)], ddof=1)

                    n0 = len(shaps[cont_var].loc[
                                 shaps[cont_var].index.isin(ref_idc)])
                    n1 = len(shaps[cont_var].loc[
                                 shaps[cont_var].index.isin(step_idc)])

                    # If there are not enough records in a group, continue the loop to the next group
                    if n0 == 0 or n1 == 0:
                        # Number label == higher
                        shap_ordinal_stats.loc['n=step', cont_var] = n1
                        # Number label == lower
                        shap_ordinal_stats.loc['n=ref', cont_var] = n0
                        continue

                    # Group mean
                    mean0 = np.mean(shaps[cont_var].loc[
                                        shaps[cont_var].index.isin(ref_idc)])
                    mean1 = np.mean(shaps[cont_var].loc[
                                        shaps[cont_var].index.isin(step_idc)])

                    # Difference group 1 - grou p0
                    difference = mean1 - mean0

                    # Welch standard error
                    se = np.sqrt(var1 / n1 + var0 / n0)

                    # Welch degrees of freedom
                    numerator = (var1 / n1 + var0 / n0) ** 2

                    denominator = (
                            (var1 / n1) ** 2 / (n1 - 1)
                            + (var0 / n0) ** 2 / (n0 - 1)
                    )

                    df = numerator / denominator

                    # 95% CI
                    alpha = 0.05
                    t_critical = t.ppf(1 - alpha / 2, df)

                    ci_lower = difference - t_critical * se
                    ci_upper = difference + t_critical * se

                    resultat = st.ttest_ind(
                        shaps[cont_var].loc[
                            shaps[cont_var].index.isin(ref_idc)],
                        shaps[cont_var].loc[
                            shaps[cont_var].index.isin(step_idc)],
                        equal_var=False,  # Welch
                        alternative="two-sided"
                    )

                    print(cont_var)
                    print(f"Number in group 0: {n0}")
                    print(f"Number in group 1: {n1}")
                    print(f"Mean group 0: {mean0:.4f}")
                    print(f"Mean group 1: {mean1:.4f}")
                    print(f"Difference, group 1 - group 0: {difference:.4f}")
                    print(f"Standard error: {se:.4f}")
                    print(f"Welch degrees of freedom: {df:.2f}")
                    print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
                    print(f"HR 95% CI: [{np.exp(difference):.4f}, {np.exp(ci_lower):.4f}, {np.exp(ci_upper):.4f}]")
                    print(f"Welchs t statistics: {resultat.statistic:.4f}")
                    print(f"Twosided p value: {resultat.pvalue:.6f}")
                    print(f"HR 95% CI: [{np.exp(difference):.2f}, {np.exp(ci_lower):.2f}, {np.exp(ci_upper):.2f}]")

                    # Add information to df
                    shap_cont_stats.loc['log_hr', cont_var] = difference
                    shap_cont_stats.loc['std', cont_var] = se
                    shap_cont_stats.loc['p-value', cont_var] = resultat.pvalue
                    shap_cont_stats.loc['hazard_ratio', cont_var] = np.exp(difference)

                    # The 95% CI
                    shap_cont_stats.loc['ci_low_lg', cont_var] = ci_lower
                    shap_cont_stats.loc['ci_high_lg', cont_var] = ci_upper
                    shap_cont_stats.loc['ci_low', cont_var] = np.exp(ci_lower)
                    shap_cont_stats.loc['ci_high', cont_var] = np.exp(ci_upper)


                ########################################################################################################

            # Change the variable name of the dataframe
            shap_cont_stats.rename(columns=change_names, inplace=True)

            # Save df for binary variables
            excel(date_time, shap_cont_stats, str(NAME) + '_shap_cont_stats', subfolder='SHAP')

            return
            # End of function

        # SHAP stats for continuous variables
        top14_con_var = ['age_at_admission', 'time_at_hospital_before_admission', 'cystatin_c', 'meds_per_pat', 'Thrombocytes', 'calprotectin', 'il6', 'laktat', 'ICU_strain', 'ferritin', 'BMI', 'systolic_blood_pressure', 'lymphocytes']
        # Run script for SHAP HR for continuous variables
        shap_stats_cont(top14_con_var, feature_mean, shap_mean, hr_ci_dataframe)

        return hr_ci_dataframe

# -------------------------------------------------------------------
# Visualize summary plots
# -------------------------------------------------------------------
def fig_summary(shapv, featv, max_disp = 12): #fig_summary(shapv, featv, max_disp = 14):

    # Top features
    plt.figure()
    beeswarm(np.array(shapv), np.array(featv), featv.columns.to_list(),
             shapv.shape, max_display=max_disp, show=False)
    save_as_svg(date_time, 'XGBoost_survival_summary_plot_' + str(max_disp), subfolder=None)

    # All features
    plt.figure()
    shap.summary_plot(np.array(shapv), featv, show=False)  # = feature_mean
    save_as_svg(date_time, 'XGBoost_survival_summary_plot_full', subfolder=None)

    return

# -------------------------------------------------------------------
# Dependency with LOESS
# -------------------------------------------------------------------
def fig_dep_loess(shap_mean, feature_mean, topfeatures):

    if len(shap_mean.columns) > 1:
        for feature in topfeatures:
            feature_name = feature # Current feature
            all_features_but_feature_name = [x for x in topfeatures if x != feature] # All other features

            # Plot dependency plots - with LOESS for continuous variables and with a circle marking reference values for the plot in the article
            plot_shap_feature(shap_mean,
                                         feature_mean,
                                         feature_name,
                                         all_features_but_feature_name,
                                         n_plots=len(topfeatures) - 1)
    return

# -------------------------------------------------------------------
# Get top features
# -------------------------------------------------------------------
def top_features(shap_mean, nbr_of_features = 14):

    # Sort features after SHAP size
    order = shap_mean.abs().mean().values.argsort()
    feature_names = shap_mean.columns.to_list()

    # Find the rank of ACEi and ARB
    descending_order = order.tolist()
    descending_order.reverse()
    for iii, idx in enumerate(descending_order):
        write_to_file(date_time, 'all_variables_rank', '\n' + str(feature_names[idx]) + ' is rank ' + str(iii) + ', mean abs SHAP value: ' + str(shap_mean[feature_names[idx]].abs().mean()))

    # Top 13
    tops = order[len(order) - nbr_of_features:].tolist()
    tops.reverse()  # get largest shap values first

    # Get all top 13 features
    topfeatures = []
    for iii, idx in enumerate(tops):
        topfeatures.append(feature_names[idx])
        write_to_file(date_time, 'variable_rank', '\n' + str(feature_names[idx]) + ' is rank ' + str(iii))

    print('Top ' + str(nbr_of_features) + ' features:', topfeatures)
    return topfeatures

# -------------------------------------------------------------------
# Kaplan-Meier and log rank test
# -------------------------------------------------------------------
def km_meds(data, variable):
    # Identify low and high medication patients and binarize into low (0) and high (1)
    df = data.copy()
    df[variable].loc[
        data[variable] < 5] = 0 # for polypharmacy
    df[variable].loc[
        data[variable] >= 5] = 1 # for polypharmacy

    df[variable] = df[variable].astype(int)

    df["time"] = data['survival_days_from_icu_start']
    df["event"] = data['mort']

    # Initialize plot
    plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter()

    # Map label
    label_map = {
        0: "< 5",
        1: "≥ 5"
    }

    # Plot survival curve for each group
    colors = ['blue','red']
    for group in df[variable].unique():
        ix = df[variable] == group
        label = label_map.get(group, str(group))  #
        kmf.fit(df.loc[ix, "time"], event_observed=df.loc[ix, "event"],
                label=label)
        kmf.plot_survival_function(color=colors[group])

    #plt.title("Kaplan-Meier curves by polypharmacy")
    plt.xlabel("Days")
    plt.ylabel("Survival probability")
    plt.grid(True)
    plt.legend(title="Number of medications per patient")
    save_as_svg(date_time, 'Kaplan_Meier_' + variable, subfolder=None)

    return

# -------------------------------------------------------------------
# Schoenfeld residuals of SHAP values
# -------------------------------------------------------------------
def schoenfeld_residuals(shap_mean, pred_and_targ, topfeatures: list):

    # Rename variables for the plots
    shap_mean.rename(columns = change_names, inplace=True)
    _, topfeatures_renamed = plot_names(change_names, 'random_word', topfeatures)

    # Create a DataFrame of SHAP values to use as covariates
    shap_prop_hazards = shap_mean[topfeatures_renamed]
    shap_prop_hazards['survival_days_from_icu_start'] = pred_and_targ['survival_days_from_icu_start']
    shap_prop_hazards['mort'] = pred_and_targ['mort']

    # Fit a CoxPH model using SHAP values as covariates
    cph = CoxPHFitter()
    cph.fit(shap_prop_hazards, duration_col='survival_days_from_icu_start', event_col='mort', show_progress=True)

    # Check proportional hazards assumption
    cph.check_assumptions(shap_prop_hazards, p_value_threshold=0.001, show_plots=True)

    return


# -------------------------------------------------------------------
# C-index at different times
# -------------------------------------------------------------------
def prop_hazards(pred_and_targ, train_pred):

    # Structured arrays
    y_train_structured = Surv.from_arrays(train_pred["mort"].astype(bool).values,
                                          train_pred["survival_days_from_icu_start"].values)
    y_test_structured = Surv.from_arrays(pred_and_targ["mort"].astype(bool).values,
                                         pred_and_targ["survival_days_from_icu_start"].values)

    # Evaluate C-index at 3, 6, 9, 12 months (in days)
    time_points = [90, 180, 270, 365]

    for t in time_points:
        c_index, _, _, _, _ = concordance_index_ipcw(
            y_train_structured,
            y_test_structured,
            pred_and_targ['validation_predictions'],
            tau=t
        )

        # CI for C-index
        val_c = []
        for re in range(n_bootstraps):
            bs_sample_val = pred_and_targ.sample(n=len(pred_and_targ),
                                                       replace=True)  # Sampling with replacement
            # Structured arrays
            y_test_structured_boot = Surv.from_arrays(bs_sample_val["mort"].astype(bool).values,
                                                 bs_sample_val["survival_days_from_icu_start"].values)

            cindex_val = concordance_index_ipcw(y_train_structured,
                                                    y_test_structured_boot,
                                                    bs_sample_val['validation_predictions'],
                                                    tau=t)
            val_c.append(cindex_val[0])

        ci_lower_val = np.percentile(val_c, 2.5, axis=0)
        ci_upper_val = np.percentile(val_c, 97.5, axis=0)

        print(f"{t} days ({t//30} months): C-index = {c_index:.3f}" + ' (' + str(round(ci_lower_val, 3)) + '-' + str(
        round(ci_upper_val, 3)) + ')')
        write_to_file(date_time, 'c-index_times', f"{t} days ({t//30} months): C-index = {c_index:.3f}" + ' (' + str(round(ci_lower_val, 3)) + '-' + str(
        round(ci_upper_val, 3)) + ')/n', 'c-index')

    return



# -------------------------------------------------------------------
# Import data files
# -------------------------------------------------------------------
results_path = r"path_to_results"
shap_val_mean =pd.read_excel(results_path + r"\mean_SHAP_values.xlsx",
                          engine='openpyxl')
feature_val = pd.read_excel(results_path + r"\feature_values.xlsx",
                          engine='openpyxl')
predictions_and_targets = pd.read_excel(results_path + r"\predictions_and_target.xlsx",
                          engine='openpyxl')
training_predictions = pd.read_excel(results_path + r"\training_predictions.xlsx",
                          engine='openpyxl')
# -------------------------------------------------------------------
# Run functions for figs and tests
# -------------------------------------------------------------------
top14features = top_features(shap_val_mean, 14)
fig_summary(shap_val_mean, feature_val, 14)
points_of_interest_df = shap_hr(shap_val_mean, feature_val, top14features, just_get_df = False)
schoenfeld_residuals(shap_val_mean, predictions_and_targets, top14features)
fig_dep_loess(shap_val_mean, feature_val, top14features)
km_meds(feature_val, "#Meds/patient")
prop_hazards(predictions_and_targets, training_predictions)

# -------------------------------------------------------------------
# Save all figs in a common PDF file
# -------------------------------------------------------------------
pdf(date_time, 'figs', None)
