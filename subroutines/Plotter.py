from qbstyles import mpl_style
import mplcyberpunk
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib as mpl
import numpy as np
import os
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import seaborn as sns
import imageio.v2 as imageio
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from ipywidgets import interact, IntSlider
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter,FuncFormatter
from matplotlib.colors import BoundaryNorm
from subroutines.Decorator import plot_style_decorator, scatter_plot_style_decorator, plot_disp_style_decorator
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.linestyle'] = '--'
plt.style.use('seaborn-v0_8-whitegrid') 
sns.set(style="whitegrid", context="talk") 



class Plotter():
    '''
    Plot the figures for training dataset, test data set, data reconstruction
    '''
    def __init__(self,y_train,y_train_pred,y_test,y_test_pred):
        '''
        Initialize the passed data

        :param y_true: origital FE output for training dataset to train a surrogate
        :param y_pred: predicted FE output using the surrogate trained
 
        '''
        self.y_train = y_train
        self.y_train_pred = y_train_pred
        self.y_test = y_test
        self.y_test_pred = y_test_pred



    @staticmethod
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

        fig, ax = plt.subplots(figsize=(4, 3), dpi = 150)

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
      



    @plot_style_decorator
    def trainSample(self): 
        '''
        Plot the some samples from training data against predicted ones
        '''
        # figure size setting, plotting style setting, 
        y_train = self.y_train 
        y_train_pred = self.y_train_pred
        

        for i in range(0,5):# sample 5 sampels
            # Add label only for the first true curve
            label_true = 'True' if i == 1 else None
            plt.plot(np.arange(0, y_train.shape[1], 1), y_train[i], label=label_true, linestyle='--', linewidth=2, color='black')

            # Add label only for the first pred curve
            label_pred = 'Predict' if i == 1 else None
            plt.plot(np.arange(0, y_train.shape[1], 1), y_train_pred[i], label=label_pred, linewidth=2, color='red')


        # Labels and title with black color
        plt.xlabel('Loading Steps', fontsize=16, color='black')
        plt.ylabel('QoI', fontsize=16, color='black')
        plt.title('Training Samples', fontsize=18, color='black')

        # Legend with black background and black text
        plt.legend(loc='upper right', fontsize=12, facecolor='black', edgecolor='black', labelcolor='black')

    @plot_style_decorator
    def testSample(self): 
        '''
        Plot the some samples from training data against predicted ones
        '''
        y_test = self.y_test 
        y_test_pred = self.y_test_pred

        for i in range(0,5):
            # Add label only for the first true curve
            label_true = 'True' if i == 1 else None
            plt.plot(np.arange(0, y_test.shape[1], 1), y_test[i], label=label_true, linestyle='--', linewidth=2, color='black')

            # Add label only for the first pred curve
            label_pred = 'Predict' if i == 1 else None
            plt.plot(np.arange(0, y_test.shape[1], 1), y_test_pred[i], label=label_pred, linewidth=2, color='red')

        # Labels and title with black color
        plt.xlabel('Loading Steps', fontsize=16, color='black')
        plt.ylabel('QoI', fontsize=16, color='black')
        plt.title('Testing Samples', fontsize=18, color='black')
        plt.tight_layout()
        plt.show()
        
    @scatter_plot_style_decorator
    def scatterTraindata(self):
        '''
        Scatter true vs pred for all training data
        '''      
        y_train = self.y_train 
        y_train_pred = self.y_train_pred
        
        def gnae(y_true, y_pred):
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)
            return np.sum(np.abs(y_pred - y_true)) / (np.sum(np.abs(y_true)) + 1e-8) *100
            
        # error metrics
        error_rate = gnae(y_train, y_train_pred)    
            
        # starting to plot scatter plot of the training set results
        sns.scatterplot(x=y_train.flatten(), y=y_train_pred.flatten(), color='#2A9D8F', alpha=0.6, edgecolor=None)
        plt.xlabel("True", fontsize=16, color='black')
        plt.ylabel("Predict", fontsize=16, color='black')
        
        all_vals = np.concatenate([y_train.flatten(), y_train_pred.flatten()])
        lims = [all_vals.min(), all_vals.max()]
        plt.plot(lims, lims, linestyle='--', color='black', linewidth=1.0)

        plt.title(f'GNAE = {error_rate:.2f}%', fontsize=16, color='black')
        
    @scatter_plot_style_decorator        
    def scatterTestdata(self):
        '''
        Scatter true vs pred for all test data
        '''        
        y_test = self.y_test 
        y_test_pred = self.y_test_pred
        def gnae(y_true, y_pred):
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)
            return np.sum(np.abs(y_pred - y_true)) / (np.sum(np.abs(y_true)) + 1e-8) *100
        # error metrics
        error_rate = gnae(y_test, y_test_pred)  
        
        # starting to plot scatter plot of the test set results
        sns.scatterplot(x=y_test.flatten(), y=y_test_pred.flatten(), color='#2A9D8F', alpha=0.6, edgecolor=None)
        plt.xlabel("True", fontsize=16, color='black')
        plt.ylabel("Predict", fontsize=16, color='black')
        
              
        all_vals = np.concatenate([y_test.flatten(), y_test_pred.flatten()])
        lims = [all_vals.min(), all_vals.max()]
        plt.plot(lims, lims, linestyle='--', color='black', linewidth=1.0)

        plt.title(f'GNAE = {error_rate:.2f}%', fontsize=16, color='black')

        
    def disp_Vector(self,y_test0,shape0_test,y_test_pred0):
        '''
        Plot a specific displacement vector sample along True vs Pred Vs Error
        '''
        # plot the heatmap for training data
        y_test0 = y_test0.reshape(shape0_test)
        y_test_pred0 = y_test_pred0.reshape(shape0_test)        
        
        # Plot the True vs pred for testing data
        @plot_disp_style_decorator
        def plot_dispV_interact(sample_idx):
            # Set the figure size and style  
            fig, ax = plt.subplots(figsize=(4, 3.5), dpi=150)
            for time_idx in range(y_test0[0].shape[0]):
                ax.plot(np.arange(0, y_test0.shape[-1], 1), y_test0[sample_idx][time_idx],  linestyle='-', linewidth=2, color="#555555")
                ax.plot(np.arange(0, y_test_pred0.shape[-1], 1), y_test_pred0[sample_idx][time_idx], linestyle='--', linewidth=2, color="#FF5733")     

            # Labels and title with black color
            ax.set_xlabel('Coordinate')
            ax.set_ylabel('Ground surface settlement (m)')
            ax.set_ylim(-0.2,0.31)
            # Create empty lines for legend
            ax.plot([], [], label='True', color='#555555', linestyle='-')
            ax.plot([], [], label='Pred', color='#FF5733', linestyle='--')
            # Legend with black background and black text
            ax.legend(loc='upper right', fontsize = 18, frameon=False)

        
        # Create interactive widgets for sample and time index selection
        w = interact(plot_dispV_interact, 
                 sample_idx=(0, y_test0.shape[0]-1))

                 
    def disp_Field(self,y_test1,y_test_pred1):
        '''
        Plot a specific displacement field sample along True vs Pred Vs Error
        '''

        # Plot the heatmap for testing data
        def plot_heatmaps(sample_idx, time_idx):
            fig = plt.figure(figsize=(15, 4),dpi = 150)
            gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])

            # color setting
            cmap_true_pred = 'cividis'   # 'plasma', 'inferno'
            cmap_error = 'RdBu_r'    
            
            vmin = y_test1[sample_idx, time_idx].min()
            vmax = y_test1[sample_idx, time_idx].max()
            #ticks = np.round(np.linspace(vmin, vmax, 6), 4) 
            ticks = np.round(np.linspace(-0.15, 0, 6), 4) 
            
            bounds = np.linspace(-0.15, 0, 7)
            norm = BoundaryNorm(boundaries=bounds, ncolors=256)

   
            ax0 = fig.add_subplot(gs[0])
            im0 = ax0.imshow(y_test1[sample_idx, time_idx], cmap=cmap_true_pred, aspect='auto', norm=norm)
            ax0.set_title(f"True: Stage {time_idx+1}",fontsize = 16)
            ax0.set_xlabel("Dim1",fontsize = 16);  ax0.set_ylabel("Dim2",fontsize = 16); 
            cbar0  = fig.colorbar(im0, ax=ax0)
            cbar0.set_ticks(ticks)  # 
            cbar0.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            cbar0.update_ticks()  # 

            # vmin = y_test_pred1[sample_idx, time_idx].min()
            # vmax = y_test_pred1[sample_idx, time_idx].max()
            # ticks = np.round(np.linspace(vmin, vmax, 6), 4) 
            
            ax1 = fig.add_subplot(gs[1])
            im1 = ax1.imshow(y_test_pred1[sample_idx, time_idx], cmap=cmap_true_pred, aspect='auto', norm=norm)
            ax1.set_title(f"Pred: Stage {time_idx+1}",fontsize = 16)
            ax1.set_xlabel("Dim1",fontsize = 16);  ax1.set_ylabel("Dim2",fontsize = 16); 
            cbar1  = fig.colorbar(im1, ax=ax1)
            cbar1.set_ticks(ticks)  # 
            cbar1.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            cbar1.update_ticks()  # 


            
            ax2 = fig.add_subplot(gs[2])
            error = y_test1[sample_idx, time_idx] - y_test_pred1[sample_idx, time_idx]
            error[0,:] = 0
            vmin = error.min()
            vmax = error.max()
            ticks = np.round(np.linspace(vmin, vmax, 6), 4) 
            
            im2 = ax2.imshow(error, cmap=cmap_error, aspect='auto', vmin=vmin, vmax=vmax) 
            ax2.set_title(f"Error (True - Pred)",fontsize = 16)
            ax2.set_xlabel("Dim1",fontsize = 16);  ax2.set_ylabel("Dim2",fontsize = 16); 
            cbar2  = fig.colorbar(im2, ax=ax2)
            cbar2.set_ticks(ticks)  # 
            cbar2.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))   
            cbar2.update_ticks()  # 
            
            plt.tight_layout()
            plt.show()

        # Create interactive widgets for sample and time index selection
        interact(plot_heatmaps, 
                 sample_idx=(0, y_test_pred1.shape[0]-1), 
                 time_idx=IntSlider(min=0, max=y_test_pred1.shape[1]-1, step=1, value=0))
                 
                 
    def disp_Field_GIF(self,y_test1,y_test_pred1,sample_idx):
        '''
        Plot a specific displacement field sample along True vs Pred Vs Error
        '''

        # Function to generate GIF for a specific sample's time evolution
        def generate_gif_for_sample(sample_idx, gif_name='heatmap_time_evolution.gif', fps=2):
            temp_dir = 'gif_frames'
            os.makedirs(temp_dir, exist_ok=True)

            for time_idx in range(y_test_pred1.shape[1]):
                fig = plt.figure(figsize=(15, 4))
                gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
                    

                # color setting
                cmap_true_pred = 'cividis'   # 'plasma', 'inferno'
                cmap_error = 'RdBu_r'    
                
                vmin = y_test1[sample_idx, time_idx].min()
                vmax = y_test1[sample_idx, time_idx].max()
                #ticks = np.round(np.linspace(vmin, vmax, 6), 4) 
                ticks = np.round(np.linspace(-0.15, 0, 6), 4) 
                
                bounds = np.linspace(-0.15, 0, 7)
                norm = BoundaryNorm(boundaries=bounds, ncolors=256)

       
                ax0 = fig.add_subplot(gs[0])
                im0 = ax0.imshow(y_test1[sample_idx, time_idx], cmap=cmap_true_pred, aspect='auto', norm=norm)
                ax0.set_title(f"True: Stage {time_idx+1}",fontsize = 16)
                ax0.set_xlabel("Dim1",fontsize = 16);  ax0.set_ylabel("Dim2",fontsize = 16); 
                cbar0  = fig.colorbar(im0, ax=ax0)
                cbar0.set_ticks(ticks)  # 
                cbar0.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                cbar0.update_ticks()  # 

                # vmin = y_test_pred1[sample_idx, time_idx].min()
                # vmax = y_test_pred1[sample_idx, time_idx].max()
                # ticks = np.round(np.linspace(vmin, vmax, 6), 4) 
                
                ax1 = fig.add_subplot(gs[1])
                im1 = ax1.imshow(y_test_pred1[sample_idx, time_idx], cmap=cmap_true_pred, aspect='auto', norm=norm)
                ax1.set_title(f"Pred: Stage {time_idx+1}",fontsize = 16)
                ax1.set_xlabel("Dim1",fontsize = 16);  ax1.set_ylabel("Dim2",fontsize = 16); 
                cbar1  = fig.colorbar(im1, ax=ax1)
                cbar1.set_ticks(ticks)  # 
                cbar1.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                cbar1.update_ticks()  # 


                
                ax2 = fig.add_subplot(gs[2])
                error = y_test1[sample_idx, time_idx] - y_test_pred1[sample_idx, time_idx]
                error[0,:] = 0
                vmin = error.min()
                vmax = error.max()
                ticks = np.round(np.linspace(vmin, vmax, 6), 4) 
                
                im2 = ax2.imshow(error, cmap=cmap_error, aspect='auto', vmin=vmin, vmax=vmax) 
                ax2.set_title(f"Error (True - Pred)",fontsize = 16)
                ax2.set_xlabel("Dim1",fontsize = 16);  ax2.set_ylabel("Dim2",fontsize = 16); 
                cbar2  = fig.colorbar(im2, ax=ax2)
                cbar2.set_ticks(ticks)  # 
                cbar2.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))   
                cbar2.update_ticks()  # 

                plt.tight_layout()
                frame_path = os.path.join(temp_dir, f"frame_{time_idx:03d}.png")
                plt.savefig(frame_path)
                plt.close()

            # generate GIF
            images = [imageio.imread(os.path.join(temp_dir, f)) 
                      for f in sorted(os.listdir(temp_dir)) if f.endswith('.png')]
            imageio.mimsave(gif_name, images, fps=fps,loop=1000)


        generate_gif_for_sample(sample_idx=sample_idx)
