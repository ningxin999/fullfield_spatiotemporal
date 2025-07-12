import numpy as np
from scipy.io import loadmat

def load_mat_data_for_dual_output(mat_path):
    """
    Load and preprocess .mat file for dual-output neural network.

    Returns:
        X: (n_samples, input_dim)
        y: list of two outputs:
            y[0]: (n_samples, timesteps, features) -- 1D vector response
            y[1]: (n_samples, timesteps, height, width) -- 2D field response
    """
    data = loadmat(mat_path, squeeze_me=True, mat_dtype=True)

    # Load data
    X = data['EDarray']
    y_raw = data['y']
    nodeLabel = data['nodeLabel']
    coord = data['coord']

    # Convert to float32 numpy arrays
    y0 = np.array(y_raw[0].tolist(), dtype=np.float32)
    y1 = np.array(y_raw[1].tolist(), dtype=np.float32)
    coord0 = np.array(coord[0], dtype=np.float32)
    coord1 = np.array(coord[1], dtype=np.float32)
    nodeLabel0 = np.array(nodeLabel[0], dtype=np.float32)
    nodeLabel1 = np.array(nodeLabel[1], dtype=np.float32)

    # Sort y0 by coord0
    idx0 = np.argsort(coord0)
    y0 = y0[:, :, idx0]

    # Sort y1 by coord1 (sort by y descending, then x)
    idx1 = np.lexsort((-coord1[:, 1], coord1[:, 0]))
    y1 = y1[:, :, idx1].reshape(y1.shape[0], y1.shape[1], 16, 21, order='F')

    # Transpose for training: (n_samples, timesteps, ...)
    y0 = np.transpose(y0, (1, 0, 2))
    y1 = np.transpose(y1, (1, 0, 2, 3))

    # Package into list
    y = [y0, y1]

    return X, y
