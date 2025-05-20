import numpy as np
import scipy.io
import os
from utils.def_function import epoch1, CCA, notch_filter, filter_2sIIR

if __name__ == '__main__':
    # Define constants
    sample_rate = 500
    N_harmonic = 4
    freqs = [13, 12, 11, 10, 9]
    notch_freq = 50
    order = 4  # Filter order
    cutoff_freq = [4, 64]
    Q = 30 #  Quality factor of the notch filter.

    # Define data paths
    data_paths = {
        'f1data': r'your path\f1data.mat',
        'f2data': r'your path\f2data.mat',
        'f3data': r'your path\f3data.mat',
        'f4data': r'your path\f4data.mat',
        'f5data': r'your path\f5data.mat'
    }

    # Load MATLAB data using scipy.io.loadmat()
    loaded_data = {}
    for key, path in data_paths.items():
        mat_data = scipy.io.loadmat(path)
        loaded_data[key] = mat_data[key]  # Extracting data_f1, data_f2, etc.

    # Initialize lists to store processed data
    processed_data = {key: [] for key in data_paths.keys()}

    # Process each dataset
    for key, data in loaded_data.items():
        for i in range(data.shape[2]):
            filtered_data_list = []
            for j in range(data.shape[1]-1):
                origin_data = np.transpose(data[:, j, i])
                bp_filtered_data = filter_2sIIR(origin_data, cutoff_freq, sample_rate, order, 'bandpass')
                nf_filtered_data = notch_filter(bp_filtered_data, sample_rate, notch_freq, Q)
                filtered_data_list.append(nf_filtered_data)
            filtered_data_concat = np.concatenate(filtered_data_list, axis=0)
            epoch_data, num_epochs, epoch_length = epoch1(filtered_data_concat, sample_rate)
            epoch_data = np.transpose(epoch_data, (2, 1, 0))
            processed_data[key].append(epoch_data)
        processed_data[key] = np.concatenate(processed_data[key], axis=0)
        processed_data[key] = np.transpose(processed_data[key], (2, 1, 0))
        processed_data[key] = processed_data[key][:, :, :]

    # Save processed data to a combined MATLAB file
    save_dir = r'your path\datasets'
    filename = 'combined_data.mat'
    file_path = os.path.join(save_dir, filename)
    scipy.io.savemat(file_path, processed_data)

    # Combine processed data into a single array
    eeg = np.zeros((processed_data['f1data'].shape[0], processed_data['f1data'].shape[1],
                    processed_data['f1data'].shape[2], len(data_paths)))

    for i, key in enumerate(data_paths.keys()):
        eeg[:, :, :, i] = processed_data[key]

    N_channel = eeg.shape[0]
    N_point = eeg.shape[1]
    N_block = eeg.shape[2]
    N_target = eeg.shape[3]

    eeg2 = np.empty((N_channel, N_point, 0))
    for i in range(N_target):
        current_target_data = eeg[:, :, :, i]
        eeg2 = np.concatenate((eeg2, current_target_data), axis=2)

    NumTrial = eeg2.shape[2]
    labels = np.concatenate([np.ones(NumTrial // 5) * freq for freq in freqs])

    # Initialize model dictionary
    model = {
        'Template': np.zeros((N_channel, N_point, N_target)),
        'Reference': np.zeros((2 * N_harmonic, N_point, N_target))
    }

    # Create time vector
    t = np.arange(1 / sample_rate, N_point / sample_rate + 1 / sample_rate, 1 / sample_rate)

    # Populate model dictionary with Template and Reference data
    for targ_i in range(N_target):
        model['Template'][:, :, targ_i] = np.mean(np.squeeze(eeg[:, :, :, targ_i]), axis=2)
        Y = np.array([])
        for har_i in range(1, N_harmonic + 1):
            Y = np.concatenate(
                (Y, np.sin(2 * np.pi * freqs[targ_i] * har_i * t), np.cos(2 * np.pi * freqs[targ_i] * har_i * t)))
        Y = Y.reshape(-1, eeg.shape[0]).T
        model['Reference'][:, :, targ_i] = Y

    # Initialize arrays to store results
    allcoeff = np.empty((0, len(freqs)))
    outputlabels = np.empty(NumTrial, dtype=int)
    Allcoefficience = np.zeros(len(freqs))

    # Perform CCA and classification
    for loocv_i in range(NumTrial):
        Testdata = eeg2[:, :, loocv_i]
        for targ_j in range(len(freqs)):
            wn1, wn2 = CCA(model['Reference'][:, :, targ_j], Testdata)
            weighted_train = wn2 @ model['Reference'][:, :, targ_j]
            weighted_test = wn1 @ Testdata
            coefficienceMatrix = np.corrcoef(weighted_test, weighted_train)
            coefficience1 = abs(coefficienceMatrix[0, 1])

            wn, _ = CCA(model['Template'][:, :, targ_j], Testdata)
            weighted_train = wn @ model['Template'][:, :, targ_j]
            weighted_test = wn @ Testdata
            coefficienceMatrix = np.corrcoef(weighted_test, weighted_train)
            coefficience2 = coefficienceMatrix[0, 1]

            wn, _ = CCA(model['Reference'][:, :, targ_j], Testdata)
            weighted_train = wn @ model['Template'][:, :, targ_j]
            weighted_test = wn @ Testdata
            coefficienceMatrix = np.corrcoef(weighted_test, weighted_train)
            coefficience3 = coefficienceMatrix[0, 1]

            wn, _ = CCA(model['Template'][:, :, targ_j], model['Reference'][:, :, targ_j])
            weighted_train = wn @ model['Template'][:, :, targ_j]
            weighted_test = wn @ Testdata
            coefficienceMatrix = np.corrcoef(weighted_test, weighted_train)
            coefficience4 = coefficienceMatrix[0, 1]

            Allcoefficience[targ_j] = abs(np.sum(
                np.sign([coefficience1, coefficience2, coefficience3, coefficience4]) * np.array(
                    [coefficience1, coefficience2, coefficience3, coefficience4]) ** 2))

        allcoeff = np.vstack((allcoeff, Allcoefficience))
        index = np.argmax(Allcoefficience)
        outputlabels[loocv_i] = freqs[index]

    trueNum = np.sum((outputlabels - labels) == 0)
    errorIndex = np.where((outputlabels - labels) != 0)[0]
    acc = trueNum / len(labels)

    # Save model data
    model['Reference'] = np.array(model['Reference'])
    model['Template'] = np.array(model['Template'])
    np.save('model_data.npy', model)

    # Output results
    print(f'\nThe number of correct predictions: {trueNum}/{NumTrial}')
    print(f'The average accuracy: {acc:.4f}')

    # Calculate Information Transfer Rate (ITR)
    Nf = 5
    Tw = 2.5
    if acc == 1:
        itr = (np.log2(Nf) + acc * np.log2(acc)) * 60 / Tw
    elif acc < 1 / Nf:
        itr = 0
    else:
        itr = (np.log2(Nf) + acc * np.log2(acc) + (1 - acc) * np.log2((1 - acc) / (Nf - 1))) * (60 / Tw)

    print(f'The ITR is: {itr:.4f} bmp')
