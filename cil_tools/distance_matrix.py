import pickle
import numpy as np
from scipy.spatial.distance import cdist


def uniform_sample_offsets(mem_size: int, num_bucket: int, sample_size: int):
    bucket_size = sample_size // num_bucket
    assert bucket_size >= mem_size

    start = int(bucket_size / 2 - mem_size / 2)
    offsets = []
    for i in range(num_bucket):
        offsets.append(start)
        start += bucket_size

    return offsets


def main():
    with open('features_from_pretrained_place365.pkl', 'rb') as f:
        data = pickle.load(f)

    mem_size = 5
    num_brackets = 8
    exemplar_brackets = [{} for _ in range(num_brackets)]
    for cls_idx in range(101):
        train_features = [record['features'] for record in data['train'][cls_idx]]
        val_features = [record['features'] for record in data['val'][cls_idx]]
        distance = cdist(train_features, val_features, metric='cosine')
        distance_to_val_set = np.min(distance, axis=1)
        sorted_indices = np.argsort(distance_to_val_set)

        offsets = uniform_sample_offsets(mem_size, num_brackets, len(sorted_indices))
        for i in range(num_brackets):
            # training_video_indices = sorted_indices[i*mem_size: (i+1)*mem_size]
            training_video_indices = sorted_indices[offsets[i]: offsets[i] + mem_size]
            exemplar_brackets[i][cls_idx] = {}
            exemplar_brackets[i][cls_idx]['records'] = [data['train'][cls_idx][i] for i in training_video_indices]
            exemplar_brackets[i][cls_idx]['mean_class_distance'] = distance_to_val_set[training_video_indices].mean()

    for i, bracket in enumerate(exemplar_brackets):
        mean_distance = np.mean([class_exemplar['mean_class_distance'] for class_exemplar in bracket.values()])
        print('bracket {} distance: {}'.format(i, mean_distance))

        with open('bracket_{}.pkl'.format(i), 'wb') as f:
            pickle.dump(bracket, f)


if __name__ == '__main__':
    main()
