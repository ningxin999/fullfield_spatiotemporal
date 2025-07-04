from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_loss(history):
    """
    Plot the training loss curves over epochs using the Keras history object.

    This function visualizes the total loss and specific output losses ('dispVector_output_loss' 
    and 'dispField_output_loss') from the training history. The losses are plotted on a logarithmic 
    scale to better display variations across different magnitudes.

    Parameters:
    -----------
    history : keras.callbacks.History
        The history object returned by model.fit(), containing loss values per epoch.

    Returns:
    --------
    None
        Displays a matplotlib plot of loss curves.
    """
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

    fig, ax = plt.subplots(figsize=(8, 5), dpi = 1200)

    ax.plot(history.history['loss'], label='Total Loss', color='black', linewidth=1.5)
    ax.plot(history.history['dispVector_output_loss'], label='Disp. Vector Loss', color='dimgray', linewidth=1.2)
    ax.plot(history.history['dispField_output_loss'], label='Disp. Field Loss', color='gray', linewidth=1.2, linestyle='--')

    ax.set_yscale('log')
    ax.grid(axis='y', which='major', linestyle='--', color='lightgray', linewidth=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(axis='both', which='major', width=1.2, length=7)
    ax.tick_params(axis='both', which='minor', width=0.8, length=4)

    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss (log scale)', fontsize=14)
    ax.set_title('Training Loss Curves', fontsize=16)
    ax.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.show()
  
