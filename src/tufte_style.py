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
        # Primary categories — aligned with Brockovich Data Center visual grammar
        'operating': '#2ECC71',   # emerald green   — operational data centers (built & running)
        'under_construction': '#E67E22',  # orange    — under construction (announced / building)
        'proposed': '#9B59B6',      # purple          — proposed (in pipeline / pending approval)
        'superfund': '#E9D502',     # blue            — superfund sites (your domain, not on Brockovich)

        # Pre/Post EO
        'pre_eo': '#A8BFDB',        # muted blue
        'post_eo': '#E8A838',       # amber

        # Regions
        'northeast': '#7B6FAB',     # purple
        'south': '#E07B54',         # warm orange
        'midwest': '#4F8EF7',       # blue
        'west': '#2ECC71',          # green

        # Neutral / supporting
        'highlight': '#E74C3C',     # red             — callout/alert
        'neutral': '#AAAAAA',       # gray            — background/reference
        'background': '#F7F7F7',    # off-white
    }

STATUS_PALETTE = {
        'Operating': COLORS['operating'],
        'Under Construction': COLORS['under_construction'],
        'Proposed': COLORS['proposed'],
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

