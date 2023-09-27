
from tf_data_pipeline.data import WaveFormData

from .qkeras_model import create_dilated_model, masked_mse

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--num-train-egs', type=int, default=200_000)
    parser.add_argument('--num-validate-egs', type=int, default=1_000)
    opts = parser.parse_args()
    print("opts", opts)

    # parse files and do splits etc
    data = WaveFormData()

    # WIP in == out == filter_size
    # TODO: do three version of qconv1d see create_dilated_model

    IN_OUT_D = 3
    NUM_LAYERS = 2
    # WIP filter size of 3; final will be 8
    FILTER_SIZE = 3

    # note: kernel size and implied dilation rate always assumed 4

    RECEPTIVE_FIELD_SIZE = 4**NUM_LAYERS
    TEST_SEQ_LEN = RECEPTIVE_FIELD_SIZE
    TRAIN_SEQ_LEN = RECEPTIVE_FIELD_SIZE * 5
    print("RECEPTIVE_FIELD_SIZE", RECEPTIVE_FIELD_SIZE)
    print("TRAIN_SEQ_LEN", TRAIN_SEQ_LEN)
    print("TEST_SEQ_LEN", TEST_SEQ_LEN)

    # make model
    train_model = create_dilated_model(TRAIN_SEQ_LEN,
            in_out_d=IN_OUT_D, num_layers=NUM_LAYERS, filter_size=FILTER_SIZE,
            all_outputs=False)

    print(train_model.summary())