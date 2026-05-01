import matplotlib.pyplot as plt

def define_plot_style():
    plt.rcParams.update(
            {
                # figure
                'figure.facecolor': 'white',
                'figure.edgecolor': 'white',
                'figure.figsize': (10, 6),
                'figure.dpi': 100,

                # axes
                'axes.facecolor': 'white',
                'axes.edgecolor': 'black',
                'axes.linewidth': 0.8,
                'axes.grid': True,
                'axes.titlesize': 14,
                'axes.titleweight': 'normal',
                'axes.labelsize': 11,
                'axes.labelweight': 'normal',
                'axes.spines.top': False,
                'axes.spines.right': False,
                # grid
                'grid.color': 'gray',
                'grid.alpha': 0.3,
                'grid.linestyle': ':',
                'grid.linewidth': 0.5,
                # ticks

                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'xtick.color': 'black',
                'ytick.color': 'black',
                'xtick.direction': 'out',
                'ytick.direction': 'out',

                # lines
                'lines.linewidth': 2,
                'lines.markersize': 6,
                # font

                'font.family': 'sans-serif',
                'font.sans-serif': ['Arial', 'DejaVu Sans'],
                'font.size': 10,

                # legend

                'legend.frameon': False,
                'legend.fontsize': 10
            }
        )


COLORS = {
        # Primary categories
        'proposed': '#E07B54',  # warm orange  — proposed data centers
        'operating': '#4F8EF7',  # blue         — operating data centers
        'superfund': '#6ABF69',  # green        — superfund sites

        # Pre/Post EO
        'pre_eo': '#A8BFDB',  # muted blue
        'post_eo': '#E8A838',  # amber

        # Regions
        'northeast': '#7B6FAB',  # purple
        'south': '#E07B54',  # warm orange
        'midwest': '#4F8EF7',  # blue
        'west': '#6ABF69',  # green

        # Neutral / supporting
        'highlight': '#E84B4B',  # red          — callout/alert
        'neutral': '#AAAAAA',  # gray         — background/reference
        'background': '#F7F7F7',  # off-white
    }

STATUS_PALETTE = {
        'Proposed': COLORS['proposed'],
        'Operating': COLORS['operating'],
    }

REGION_PALETTE = {
        'Northeast': COLORS['northeast'],
        'South': COLORS['south'],
        'Midwest': COLORS['midwest'],
        'West': COLORS['west'],
    }

EO_PALETTE = {
        False: COLORS['pre_eo'],
        True: COLORS['post_eo'],
    }
