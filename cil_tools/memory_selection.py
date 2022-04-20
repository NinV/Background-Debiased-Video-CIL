import argparse
import json
import numpy as np
from tqdm import tqdm

"""
 def _construct_exemplar_unified(self, data_manager, m):
        logging.info('Constructing exemplars for new classes...({} per classes)'.format(m))
        _class_means = np.zeros((self._total_classes, self.feature_dim))

        # Calculate the means of old classes with newly trained network
        for class_idx in range(self._known_classes):
            mask = np.where(self._targets_memory == class_idx)[0]
            class_data, class_targets = self._data_memory[mask], self._targets_memory[mask]

            class_dset = data_manager.get_dataset([], source='train', mode='test',
                                                  appendent=(class_data, class_targets))
            class_loader = DataLoader(class_dset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        # Construct exemplars for new classes and calculate the means
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, class_dset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                 mode='test', ret_data=True)
            class_loader = DataLoader(class_dset, batch_size=batch_size, shuffle=False, num_workers=4)

            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []
            for k in range(1, m+1):
                S = np.sum(exemplar_vectors, axis=0)  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                selected_exemplars.append(np.array(data[i]))  # New object to avoid passing by inference
                exemplar_vectors.append(np.array(vectors[i]))  # New object to avoid passing by inference

                vectors = np.delete(vectors, i, axis=0)  # Remove it to avoid duplicative selection
                data = np.delete(data, i, axis=0)  # Remove it to avoid duplicative selection

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = np.concatenate((self._data_memory, selected_exemplars)) if len(self._data_memory) != 0 \
                else selected_exemplars
            self._targets_memory = np.concatenate((self._targets_memory, exemplar_targets)) if \
                len(self._targets_memory) != 0 else exemplar_targets

            # Exemplar mean
            exemplar_dset = data_manager.get_dataset([], source='train', mode='test',
                                                     appendent=(selected_exemplars, exemplar_targets))
            exemplar_loader = DataLoader(exemplar_dset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(exemplar_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        self._class_means = _class_means
© 2022 GitHub, Inc.
Terms
Privacy
Security


"""


"""
Generally, cluster validity measures are categorized into 3 classes, they are – 
 

Internal cluster validation : The clustering result is evaluated based on the data clustered itself 
(internal information) without reference to external information.

External cluster validation : Clustering results are evaluated based on some externally known result, 
such as externally provided class labels.

Relative cluster validation : The clustering results are evaluated by varying different parameters for the same algorithm (e.g. changing the number of clusters).

"""

def parse_args():
    # config_file = 'configs/cil/tsm/tsm_r34_1x1x8_25e_ucf101_rgb_task_0.py'
    # ckpt_file = 'work_dirs/tsm_r34_1x1x8_25e_ucf101_hflip_rgb_task_0/epoch_50.pth'
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')
    parser.add_argument('--size', type=int, default=5)
    return parser.parse_args()


class CHI:
    """
    Calinski-Harabasz Index
    """
    def __init__(self):
        pass

    def __call__(self, ):

def main():
    args = parse_args()
    with open(args.json_file, 'r') as f:
        data = json.load(f)

    num_classes = len(data.keys())

    exemplar = {}
    for cls_id, sample_list in data.items():
        exemplar[cls_id] = []
        sample_dict = {s['frame_dir']: [s['cls_score'], s['repr_consensus'], s['total_frames'], s['label']] for s in sample_list}

        similarity_score = float('-inf')
        while len(exemplar[cls_id]) < args['size']:
            for sample_frame_dir, [cls_score, repr_consensus, total_frames, label] in sample_dict.items():
                score = similarity_score(exemplar[cls_id])


        cls_score = sa
        # if torch.argmax(cls_score).item() == sample_info['label']:
        #     sample_info['cls_score'] = cls_score.tolist()
        #     sample_info['repr_consensus'] = repr_consensus.tolist()
        #     try:
        #         features_by_class[sample_info['label']].append(sample_info)
        #     except KeyError:
        #         features_by_class[sample_info['label']] = [sample_info]




if __name__ == '__main__':
    main()