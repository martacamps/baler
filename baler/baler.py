import time
import os
import pandas as pd

from modules import helper


def main():
    input_path, output_path, model_path, config, mode = helper.get_arguments()
    if mode == 'train':
        perform_training()
    elif mode == 'plot':
        helper.plot(input_path, output_path)
        helper.loss_plotter('projects/cms/output/loss_data.csv', output_path)
    elif mode == 'compress':
        perform_compression()
    elif mode == 'decompress':
        perform_decompression()
    elif mode == 'info':
        perform_testing()


def perform_training():
    input_path, output_path, model_path, config, mode = helper.get_arguments()

    # ----------------------------------------------------------------------
    # Retrieve features

    train_set, test_set, number_of_columns, normalization_features = \
        helper.process(input_path, config)

    # ----------------------------------------------------------------------
    # Perform normalisation

    train_set_norm = helper.normalize(train_set, config)
    test_set_norm = helper.normalize(test_set, config)

    # ---------------------------------------------------------------------
    # Initialise model object

    ModelObject = helper.model_init(config=config)
    model = ModelObject(n_features=number_of_columns,
                        z_dim=config['latent_space_size'])

    # ---------------------------------------------------------------------
    # Perform training

    test_data_tensor, reconstructed_data_tensor = \
        helper.train(model,
                     number_of_columns,
                     train_set_norm,
                     test_set_norm,
                     output_path,
                     config)
    test_data = helper.detach(test_data_tensor)
    reconstructed_data = helper.detach(reconstructed_data_tensor)

    # ---------------------------------------------------------------------
    # Perform unnormalisation

    # Begin timing
    print('Un-normalzing...')
    start = time.time()

    # Unnormalise
    test_data_renorm = \
        helper.renormalize(test_data,
                           normalization_features['True min'],
                           normalization_features['Feature Range'],
                           config)
    reconstructed_data_renorm = \
        helper.renormalize(reconstructed_data,
                           normalization_features['True min'],
                           normalization_features['Feature Range'],
                           config)

    # End timing
    end = time.time()
    print(f'Un-normalization took:{(end - start) / 60:.3} minutes')

    # ---------------------------------------------------------------------
    # Creatre pickles and save to disc

    helper.to_pickle(test_data_renorm, f'{output_path}before.pickle')
    helper.to_pickle(reconstructed_data_renorm, f'{output_path}after.pickle')
    normalization_features.to_csv(f'{output_path}cms_normalization_features.csv')
    helper.model_saver(model, f'{output_path}current_model_15.pt')


def perform_compression():
    input_path, output_path, model_path, config, mode = helper.get_arguments()

    # ----------------------------------------------------------------------
    # Perform compression

    # Begin timing
    start = time.time()

    # Compression
    print('Compressing...')
    compressed, data_before = \
        helper.compress(model_path=model_path,
                        number_of_columns=config['number_of_columns'],
                        input_path=input_path, config=config)

    # Converting back to numpyarray
    compressed = helper.detach(compressed)

    # End timing
    end = time.time()
    print(f'Compression took: {(end - start) / 60:.3} minutes')

    # ---------------------------------------------------------------------
    # Creatre pickles and save to disc

    helper.to_pickle(compressed, f'{output_path}compressed.pickle')
    helper.to_pickle(data_before, f'{output_path}cleandata_pre_comp.pickle')


def perform_decompression():
    input_path, output_path, model_path, config, mode = helper.get_arguments()

    # ----------------------------------------------------------------------
    # Perform decompression

    # Begin timing
    start = time.time()

    # Decompression
    print('Decompressing...')
    decompressed = \
        helper.decompress(model_path=model_path,
                          number_of_columns=config['number_of_columns'],
                          input_path=input_path,
                          config=config)

    # Converting back to numpyarray
    decompressed = helper.detach(decompressed)
    normalization_features = pd.read_csv(f'{output_path}cms_normalization_features.csv')
    decompressed = \
        helper.renormalize(decompressed,
                           normalization_features['True min'],
                           normalization_features['Feature Range'],
                           config)

    # End timing
    end = time.time()
    print(f'Decompression took: {(end - start) / 60:.3} minutes')

    # ---------------------------------------------------------------------
    # Creatre pickles and save to disc

    if config['save_as_root']:
        helper.to_root(decompressed, config, f'{output_path}decompressed.root')
        helper.to_pickle(decompressed, f'{output_path}decompressed.pickle')
    else:
        helper.to_pickle(decompressed, f'{output_path}decompressed.pickle')


def perform_testing():
    print(''' ==========================
This is a mode for testing
========================== ''')

    pre_compression = 'projects/cms/output/cleandata_pre_comp.pickle'
    compressed = 'projects/cms/output/compressed.pickle'
    decompressed = 'projects/cms/output/decompressed.pickle'

    files = [pre_compression, compressed, decompressed]
    q = []
    for filename in files:
        q.append(os.stat(filename).st_size / (1024*1024))

    print(f'File size before compression: {round(q[0],2)} MB')
    print(f'Compressed file size: {round(q[1],2)} MB')
    print(f'De-compressed file size: {round(q[2],2)} MB')
    print(f'Compression ratio: {round(q[0]/q[1],2)}')
    print(f'Compressed file is {round(q[1]/q[0],2)*100}% the size of the original')


if __name__ == '__main__':
    main()
