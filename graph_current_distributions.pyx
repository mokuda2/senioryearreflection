import numpy as np
import matplotlib.pyplot as plt
import os

def graph_distributions(dict target_marriage_probs, dict model_marriage_probs, dict adjusted_marriage_probs, int gen_num, str name, str outpath, bint save_plots=True, double alpha=0.85, double eps=1e-7):
    cdef list model_vals = list(model_marriage_probs.values())
    cdef list target_vals = list(target_marriage_probs.values())
    cdef list adjusted_vals = list(adjusted_marriage_probs.values())

    cdef double[:] target_vals_np = np.array(target_vals, dtype=np.double)
    cdef double[:] model_vals_np = np.array(model_vals, dtype=np.double)
    cdef double[:] adjusted_vals_np = np.array(adjusted_vals, dtype=np.double)

    cdef int i, n

    # Create empty lists to store the filtered values
    cdef list target_vals_filtered = []
    cdef list model_vals_filtered = []
    cdef list adjusted_vals_filtered = []

    # Get the length of the arrays
    n = target_vals_np.shape[0]

    # Iterate over the elements and filter based on inequality
    for i in range(n):
        if target_vals_np[i] != 0:
            target_vals_filtered.append(target_vals_np[i])

    for i in range(n):
        if model_vals_np[i] != 0:
            model_vals_filtered.append(model_vals_np[i])

    for i in range(n):
        if adjusted_vals_np[i] != 0:
            adjusted_vals_filtered.append(adjusted_vals_np[i])

    # Convert the filtered lists back to typed memory views
    cdef double[:] target_vals_filtered_np = np.array(target_vals_filtered, dtype=np.double)
    cdef double[:] model_vals_filtered_np = np.array(model_vals_filtered, dtype=np.double)
    cdef double[:] adjusted_vals_filtered_np = np.array(adjusted_vals_filtered, dtype=np.double)

    cdef double target_eps = np.min(target_vals_filtered)
    cdef double model_eps = np.min(model_vals_filtered)
    cdef double adjusted_eps = np.min(adjusted_vals_filtered)

    cdef list target_distances, model_distances, adjusted_distances

    if eps != 0:
        target_distances = [k for k in target_marriage_probs.keys() if target_marriage_probs[k] > target_eps]
        model_distances = [k for k in model_marriage_probs.keys() if model_marriage_probs[k] > model_eps ]
        adjusted_distances = [k for k in adjusted_marriage_probs.keys() if adjusted_marriage_probs[k] > adjusted_eps]
    else:
        target_distances = [k for k in target_marriage_probs.keys() if target_marriage_probs[k] >= target_eps]
        model_distances = [k for k in model_marriage_probs.keys() if model_marriage_probs[k] >= model_eps ]
        adjusted_distances = [k for k in adjusted_marriage_probs.keys() if adjusted_marriage_probs[k] >= adjusted_eps]

    cdef int max_bin = int(max(model_distances + target_distances + adjusted_distances))

    cdef double width = 0.3
    cdef double[:] target_dist_values = [target_marriage_probs[k] for k in target_marriage_probs.keys() if k <= max_bin]
    cdef double[:] model_dist_values = [model_marriage_probs[k] for k in model_marriage_probs.keys() if k <= max_bin]
    cdef double[:] adjusted_dist_values = [adjusted_marriage_probs[k] for k in adjusted_marriage_probs.keys() if k <= max_bin]

    cdef double[:] target_dist_keys = [k for k in target_marriage_probs.keys() if k <= max_bin]
    cdef double[:] model_dist_keys = [k + width for k in model_marriage_probs.keys() if k <= max_bin]
    cdef double[:] adjusted_dist_keys = [k + 2 * width for k in adjusted_marriage_probs.keys() if k <= max_bin]

    cdef double[:] target_dist_keys_np = np.array(target_dist_keys, dtype=np.double)
    cdef double[:] model_dist_keys_np = np.array(model_dist_keys, dtype=np.double)
    cdef double[:] adjusted_dist_keys_np = np.array(adjusted_dist_keys, dtype=np.double)

    fig = plt.figure(figsize=(12,9), dpi=300)
    plt.bar(target_dist_keys_np, target_dist_values, alpha=alpha, label='target', width=width, align='edge')
    plt.bar(model_dist_keys_np, model_dist_values, alpha=alpha, label='model', width=width, align='edge')
    plt.bar(adjusted_dist_keys_np, adjusted_dist_values, alpha=alpha, label='adjusted', width=width, align='edge')

    plt.legend()
    title = name + '\n'
    title += f'generation: {gen_num} \n'

    plt.title(title, fontsize=12, pad=2)
    if not save_plots:
        plt.show()
    else:
        plt.savefig(os.path.join(outpath,  name + f'distributions_generation_{gen_num}' + '.png'), format='png')
    plt.clf()
    plt.close(fig)
