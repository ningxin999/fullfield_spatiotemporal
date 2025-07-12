from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import functools
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MaxNLocator
plt.style.use('seaborn-v0_8-whitegrid') 
sns.set(style="whitegrid", context="talk") 




def plot_style_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        plt.figure(figsize=(1, 1), dpi=150)
        ax = plt.gca()
        fig = plt.gcf()

        # Set black background
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')

        # Hide top and right spines, set bottom and left spine colors
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')

        # Tick parameters
        ax.tick_params(axis='x', colors='black', labelsize=14)
        ax.tick_params(axis='y', colors='black', labelsize=14)



        # Legend styling
        plt.legend(loc='upper right', fontsize=12, facecolor='black', edgecolor='black', labelcolor='black')

        # Call the original plotting function
        result = func(*args, **kwargs)

        plt.tight_layout()
        plt.show()

        return result
    return wrapper


def scatter_plot_style_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Apply unified matplotlib rcParams
        mpl.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 14,
            "axes.linewidth": 1.2,
            "xtick.major.size": 7,
            "xtick.minor.size": 4,
            "ytick.major.size": 7,
            "ytick.minor.size": 4,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": False,
        })
        
        # Create figure here, so style is applied before plotting
        plt.figure(figsize=(3, 3), dpi=150)
        
        # Call your original plotting function
        result = func(*args, **kwargs)
        
        # Apply final styling adjustments
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(rotation=45, fontsize=12)
        plt.axis('equal')

        # Limit number of ticks on axes
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=6))
        plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6))

        plt.tight_layout()
        plt.show()
        
        return result
    return wrapper
    
def plot_disp_style_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        mpl.rcParams.update({
            "font.family": "serif",
            "font.size": 14,
            "axes.linewidth": 1.2,
            "axes.edgecolor": "black",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "lines.markersize": 6,
        })

        

        # Pass ax to the original function so it can plot on it
        result = func(*args, **kwargs)

        plt.tight_layout()
        plt.show()

        return result
    return wrapper