import numpy as np

#Random Forest
def _weighted_average_leaf_depth(tree):
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    n_node_samples = tree.tree_.n_node_samples

    stack = [(0, 0)]
    weighted_depth_sum = 0.0
    total_leaf_samples = 0.0

    while stack:
        node_id, depth = stack.pop()
        left = children_left[node_id]
        right = children_right[node_id]

        is_leaf = left == -1 and right == -1
        if is_leaf:
            weight = float(n_node_samples[node_id])
            weighted_depth_sum += depth * weight
            total_leaf_samples += weight
            continue

        stack.append((left, depth + 1))
        stack.append((right, depth + 1))

    if total_leaf_samples == 0:
        return 0.0

    return weighted_depth_sum / total_leaf_samples


def flops_rf_per_sample(model):
    n_estimators = len(model.estimators_)
    n_classes = len(model.classes_)

    avg_depth = np.mean([_weighted_average_leaf_depth(est) for est in model.estimators_])

    # Node comparisons along paths + class-score accumulation + final argmax
    comparisons = n_estimators * avg_depth
    vote_accumulation = n_estimators * n_classes
    final_decision = n_classes - 1

    return int(round(comparisons + vote_accumulation + final_decision))

#Logistic Regression
def flops_logreg_per_sample(d, n_classes=2):
    #d multiplies + (d - 1) adds per class
    linear = n_classes * (2 * d - 1)

    if n_classes == 2:
        # Sigmoid + threshold
        activation = 5
    else:
        # Softmax (exp + sum + divide) + argmax
        activation = (3 * n_classes - 1) + (n_classes - 1)

    return linear + activation

#SVM with linear kernel
def flops_linearsvc_per_sample(d, n_classes=2):
    if n_classes <= 2:
        n_hyperplanes = 1
    else:
        n_hyperplanes = n_classes

    linear = n_hyperplanes * (2 * d - 1)
    decision = 1 if n_classes <= 2 else (n_classes - 1)

    return linear + decision

#K-Nearest Neighbors
def flops_knn_per_sample(model, d):
    n_train = model._fit_X.shape[0]
    k = model.n_neighbors
    n_classes = len(model.classes_)

    #Brute-force squared Euclidean distance to all train points:
    #d subtract + d multiply + (d - 1) add per train sample
    distance_cost = n_train * (3 * d - 1)

    #Select k smallest distances
    selection_cost = int(round(n_train * np.log2(max(k, 2))))

    #Majority vote among k labels + final argmax
    vote_cost = k + (n_classes - 1)

    return int(distance_cost + selection_cost + vote_cost)

def flops_standard_scaler(d):
    return 2 * d

def flops_maxabs_scaler(d):
    return d

