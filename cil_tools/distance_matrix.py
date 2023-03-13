import pickle
import numpy as np
from scipy.spatial.distance import cdist


def topk(a, k):
    """
    http://seanlaw.github.io/2020/01/03/finding-top-or-bottom-k-in-a-numpy-array/
    """
    return np.argpartition(a, -k)[-k:]


def botk(a, k):
    return np.argpartition(a, k)[:k]


def main():
    with open('features_from_pretrained_place365.pkl', 'rb') as f:
        data = pickle.load(f)

    k = 5
    min_distance_set = {}
    max_distance_set = {}
    for cls_idx in range(101):
        min_distance_set[cls_idx] = {}
        max_distance_set[cls_idx] = {}

        train_features = [record['features'] for record in data['train'][cls_idx]]
        val_features = [record['features'] for record in data['val'][cls_idx]]
        distance = 1 - cdist(train_features, val_features, metric='cosine')
        distance_to_val_set = np.min(distance, axis=1)

        topk_indices = topk(distance_to_val_set, k)
        max_distance_set[cls_idx]['records'] = [data['train'][cls_idx][i] for i in topk_indices]
        max_distance_set[cls_idx]['mean_distance'] = distance_to_val_set[topk_indices].mean()

        botk_indices = botk(distance_to_val_set, k)
        min_distance_set[cls_idx]['records'] = [data['train'][cls_idx][i] for i in botk_indices]
        min_distance_set[cls_idx]['mean_distance'] = distance_to_val_set[botk_indices].mean()

    with open('min_distance_set.pkl', 'wb') as f:
        pickle.dump(min_distance_set, f)

    with open('max_distance_set.pkl', 'wb') as f:
        pickle.dump(max_distance_set, f)


if __name__ == '__main__':
    main()
