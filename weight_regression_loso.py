import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from deep_model import DeepModel

def load_weight_wise(norm_raw, cuts):
    weights = norm_raw[0].tolist()
    weight_cuts = cuts[0].tolist()
    for subject,cut in zip(norm_raw[1:],cuts[1:]):
        for weight,_ in enumerate(weights):
            weights[weight].extend(subject[weight])
            weight_cuts[weight].extend(cut[weight])
    return weights, weight_cuts


def random_window_split(input_3part_emg, window_size=250, number_of_splits=50, select_part=1, select_muscle=range(6)):
    splits = np.zeros((number_of_splits, window_size, len(select_muscle)))
    emg = input_3part_emg[select_part]
    random_points = np.random.choice(range(0, len(emg)-window_size), number_of_splits,replace=False)
    for _, random_index in enumerate(random_points):
        splits[_] = emg[random_index: random_index+window_size, select_muscle]
    return splits




if __name__ == '__main__':

    train_norm_raw = np.load('data_npy/norm_raw_dataset.npy')[[0,2,3,4]]
    train_cuts = np.reshape(np.load('data_npy/cuts.npy'), (5, 6, 5, -1, 2))[[0,2,3,4]]

    test_norm_raw = np.load('data_npy/norm_raw_dataset.npy')[[1,]]
    test_cuts = np.reshape(np.load('data_npy/cuts.npy'), (5, 6, 5, -1, 2))[[0,]]

    train_weights , train_weight_cuts = load_weight_wise(train_norm_raw, train_cuts)

    train_sample = np.reshape(np.array(train_weights), (-1, 15000, 14))

    train_cut = np.reshape(np.array(train_weight_cuts), (-1, 3, 2))

    train_label = np.reshape([[1]*20,[2]*20,[3]*20,[4]*20,[5]*20,[6]*20], -1)


    test_weights , test_weight_cuts = load_weight_wise(test_norm_raw, test_cuts)

    test_sample = np.reshape(np.array(test_weights), (-1, 15000, 14))

    test_cut = np.reshape(np.array(test_weight_cuts), (-1, 3, 2))

    test_label = np.reshape([[1]*5,[2]*5,[3]*5,[4]*5,[5]*5,[6]*5], -1)

    # shuffler = range(np.shape(train_sample)[0])
    # np.random.shuffle(shuffler)
    #
    # test_index = np.random.choice(shuffler, size=10, replace=False)
    # train_index = [x for x in shuffler if x not in test_index]

    number_of_splits = 100

    segmented_train_sample = []
    segmented_train_label = []


    for sample, label, cuts in zip(train_sample, train_label, train_cut):
        trial_activity = []
        for cut in cuts:
            trial_activity.append(sample[int(cut[0]):int(cut[1])])
        segmented_train_sample.extend(random_window_split(trial_activity, number_of_splits=number_of_splits))
        # segmented_train_label.extend([np.eye(6)[label-1]] * number_of_splits) # to classification
        segmented_train_label.extend([label] * number_of_splits) # to regression


    segmented_test_sample = []
    segmented_test_label = []

    for sample, label, cuts in zip(test_sample, test_label, test_cut):
        trial_activity = []
        for cut in cuts:
            trial_activity.append(sample[int(cut[0]):int(cut[1])])
        segmented_test_sample.extend(random_window_split(trial_activity, number_of_splits=number_of_splits))
        # segmented_test_label.extend([np.eye(6)[label-1]] * number_of_splits)# to classification
        segmented_test_label.extend([label] * number_of_splits) # to regression


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
