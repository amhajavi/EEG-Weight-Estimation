import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from deep_model import DeepModel

train_split = 15
test_split = 25-train_split

def load_weight_wise(norm_raw, cuts):
    weights = norm_raw[0].tolist()
    weight_cuts = cuts[0].tolist()
    for subject,cut in zip(norm_raw[1:],cuts[1:]):
        for weight,_ in enumerate(weights):
            weights[weight].extend(subject[weight])
            weight_cuts[weight].extend(cut[weight])
    return weights, weight_cuts


def random_window_split(input_3part_emg, window_size=250, number_of_splits=50, select_part=2, select_muscle=range(6)):
    splits = np.zeros((number_of_splits, window_size, len(select_muscle)))
    emg = input_3part_emg[select_part]
    random_points = np.random.choice(range(0, len(emg)-window_size), number_of_splits,replace=False)
    for _, random_index in enumerate(random_points):
        splits[_] = emg[random_index: random_index+window_size, select_muscle]
    return splits




if __name__ == '__main__':
    norm_raw = [np.load('data_npy_old/norm_rms_dataset.npy')[0]]
    cuts = [np.reshape(np.load('data_npy/cuts.npy'), (5, 6, 5, -1, 2))[0]]
    weights , weight_cuts = load_weight_wise(norm_raw, cuts)

    whole_sample = np.reshape(np.array(weights), (-1, 15000, 14))

    whole_cut = np.reshape(np.array(weight_cuts), (-1, 3, 2))

    whole_label = np.reshape([[1]*25,[2]*25,[3]*25,[4]*25,[5]*25,[6]*25], -1)

    shuffler = range(np.shape(whole_sample)[0])
    np.random.shuffle(shuffler)

    test_index = np.random.choice(shuffler, size=10, replace=False)
    train_index = [x for x in shuffler if x not in test_index]

    number_of_splits = 100

    segmented_train_sample = []
    segmented_train_label = []


    for sample, label, cuts in zip(whole_sample[train_index], whole_label[train_index], whole_cut[train_index]):
        trial_activity = []
        for cut in cuts:
            trial_activity.append(sample[int(cut[0]):int(cut[1])])
        segmented_train_sample.extend(random_window_split(trial_activity, number_of_splits=number_of_splits))
        segmented_train_label.extend([label] * number_of_splits)


    segmented_test_sample = []
    segmented_test_label = []

    for sample, label, cuts in zip(whole_sample[test_index], whole_label[test_index], whole_cut[test_index]):
        trial_activity = []
        for cut in cuts:
            trial_activity.append(sample[int(cut[0]):int(cut[1])])
        segmented_test_sample.extend(random_window_split(trial_activity, number_of_splits=number_of_splits))
        segmented_test_label.extend([label] * number_of_splits)


    model = DeepModel()

    model.train(
                [segmented_train_sample],
                [segmented_train_label],
                [segmented_test_sample],
                [segmented_test_label])

    # clf = svm.SVC(gamma='scale', probability=True)
    # clf.fit(segmented_train_sample, segmented_train_label)
    # predicted = clf.predict(segmented_test_sample)
    # print(accuracy_score(segmented_test_label, predicted))
