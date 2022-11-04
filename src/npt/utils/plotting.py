from matplotlib import rcParams
from matplotlib import rc
import seaborn as sns

""""Set some global params and plotting settings."""

# Font options
rc('text', usetex=True)
# rcParams['pdf.fonttype'] = 42
# rcParams['ps.fonttype'] = 42
fs = 9
label_fs = fs - 1
family = 'serif'
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Times']
rcParams['font.size'] = fs

prop = dict(size=fs)
legend_kwargs = dict(frameon=True, prop=prop)
new_kwargs = dict(prop=dict(size=fs-4))

# Styling
c = 'black'
rcParams.update({'axes.edgecolor': c, 'xtick.color': c, 'ytick.color': c})
rcParams.update({'axes.linewidth': 0.5})
linewidth = 3.25063  # in inches
textwidth = 6.75133

# Global Names (Sadly not always used)
acquisition_step_label = 'Acquired Points'
LABEL_ACQUIRED_DOUBLE = 'Acquired Points'
LABEL_ACQUIRED_FULL = 'Number of Acquired Test Points'
diff_to_empircal_label = r'Difference to Full Test Loss'
std_diff_to_empirical_label = 'Standard Deviation of Estimator Error'
sample_efficiency_label = 'Efficiency'
LABEL_RANDOM = 'I.I.D. Acquisition'
LABEL_STD = 'Median \n Squared Error'
LABEL_RELATIVE_COST = 'Relative Labeling Cost'
LABEL_MEAN_LOG = 'Mean Log Squared Error'
# Color palette
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
CB_color_cycle = [CB_color_cycle[i] for i in [0, 1, 2, -2, 5, 4, 3, -3, -1]]
cbpal = sns.palettes.color_palette(palette=CB_color_cycle)
pal = sns.color_palette('colorblind')
pal[5], pal[6], pal[-2] = cbpal[5], cbpal[6], cbpal[-1]