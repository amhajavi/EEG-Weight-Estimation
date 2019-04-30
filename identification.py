import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from deep_model_identification import DeepModel




def random_window_split(input_3part_emg, window_size=250, number_of_splits=50, select_part=1, select_muscle=range(6)):
    splits = np.zeros((number_of_splits, window_size, len(select_muscle)))
    emg = input_3part_emg[select_part]
    random_points = np.random.choice(range(0, len(emg)-window_size), number_of_splits,replace=False)
    for _, random_index in enumerate(random_points):
        splits[_] = emg[random_index: random_index+window_size, select_muscle]
    return splits




if __name__ == '__main__':

    norm_raw = np.load('data_npy_old/norm_raw_dataset.npy')

    whole_sample = np.reshape(np.array(norm_raw), (-1, 15000, 14))
    whole_label = np.reshape([[0]*30,[1]*30,[2]*30,[3]*30,[4]*30], -1)
    whole_cuts = np.reshape(np.load('data_npy/cuts.npy'), (-1, 3, 2))

    shuffler = range(np.shape(whole_sample)[0])
    np.random.shuffle(shuffler)
    test_index = np.random.choice(shuffler, size=10, replace=False)
    train_index = [x for x in shuffler if x not in test_index]

    number_of_splits = 10

    segmented_train_sample = []
    segmented_train_label = []


    for sample, label, cuts in zip(whole_sample[train_index], whole_label[train_index], whole_cuts[train_index]):
        trial_activity = []
        for cut in cuts:
            trial_activity.append(sample[int(cut[0]):int(cut[1])])
        segmented_train_sample.extend(random_window_split(trial_activity, number_of_splits=number_of_splits))
        segmented_train_label.extend([np.eye(5)[label]] * number_of_splits)

    segmented_test_sample = []
    segmented_test_label = []

    for sample, label, cuts in zip(whole_sample[test_index], whole_label[test_index], whole_cuts[test_index]):
        trial_activity = []
        for cut in cuts:
            trial_activity.append(sample[int(cut[0]):int(cut[1])])
        segmented_test_sample.extend(random_window_split(trial_activity, number_of_splits=number_of_splits))
        segmented_test_label.extend([np.eye(5)[label]] * number_of_splits)


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
