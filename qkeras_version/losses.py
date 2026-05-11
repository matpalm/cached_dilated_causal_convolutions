import tensorflow as tf


def masked_mse(receptive_field_size, filter_column_idx=None):
    """
    Calculates masked version of mean square error

    Parameters:
        receptive_field_size: the number of leading elements to ignore in loss as
                              these are "polluted" by the 0 padding during training.
        filter_column_idx: only calculate loss w.r.t this column in output. done since
                           the output has 4 outs, but we might only care about one.
    Returns:
        keras loss function
    """

    def loss_fn(y_true, y_pred):
        assert len(y_true.shape) == 3, "expected (batch, sequence_length, output_dim)"
        if filter_column_idx is not None:
            # consider only a single column from output for loss
            y_true = y_true[:, :, filter_column_idx : filter_column_idx + 1]
            y_pred = y_pred[:, :, filter_column_idx : filter_column_idx + 1]
        assert y_true.shape == y_pred.shape
        # average over elements of y
        mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
        # we want to ignore the first elements of the loss since they
        # have been fed with left padded data
        mse = mse[:, receptive_field_size:]
        # return average over batch and sequence
        return tf.reduce_mean(mse)

    return loss_fn


def masked_multires_stft_loss(
    receptive_field_size,
    filter_column_idx=None,
    fft_sizes=(64, 128, 256),
    hop_sizes=(16, 32, 64),
    win_lengths=(64, 128, 256),
    w_time=0.5,
    w_mag=0.4,
    w_sc=0.1,
    eps=1e-6,
):
    assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)

    def _stft_mag(x, fft_size, hop, win):
        s = tf.signal.stft(
            x,
            frame_length=win,
            frame_step=hop,
            fft_length=fft_size,
            window_fn=tf.signal.hann_window,
            pad_end=False,
        )
        return tf.abs(s)

    def loss_fn(y_true, y_pred):
        # y: (batch, seq, channels)
        if filter_column_idx is not None:
            y_true_ = y_true[:, :, filter_column_idx : filter_column_idx + 1]
            y_pred_ = y_pred[:, :, filter_column_idx : filter_column_idx + 1]
        else:
            y_true_ = y_true
            y_pred_ = y_pred

        # mask left-padded receptive field
        y_true_ = y_true_[:, receptive_field_size:, :]
        y_pred_ = y_pred_[:, receptive_field_size:, :]

        # collapse channel dim for STFT (assuming 1 selected channel)
        y_true_1d = tf.squeeze(y_true_, axis=-1)
        y_pred_1d = tf.squeeze(y_pred_, axis=-1)

        time_mse = tf.reduce_mean(tf.square(y_true_1d - y_pred_1d))

        mr_mag = 0.0
        mr_sc = 0.0
        n = float(len(fft_sizes))

        for fft_size, hop, win in zip(fft_sizes, hop_sizes, win_lengths):
            m_true = _stft_mag(y_true_1d, fft_size, hop, win)
            m_pred = _stft_mag(y_pred_1d, fft_size, hop, win)

            # log-mag L1 is usually more perceptual than linear-mag MSE
            log_true = tf.math.log(m_true + eps)
            log_pred = tf.math.log(m_pred + eps)
            mr_mag += tf.reduce_mean(tf.abs(log_true - log_pred))

            # spectral convergence
            num = tf.norm(m_true - m_pred, ord="euclidean", axis=[-2, -1])
            den = tf.norm(m_true, ord="euclidean", axis=[-2, -1]) + eps
            mr_sc += tf.reduce_mean(num / den)

        mr_mag /= n
        mr_sc /= n

        return w_time * time_mse + w_mag * mr_mag + w_sc * mr_sc

    return loss_fn


def combined_masked_loss(
    receptive_field_size,
    filter_column_idx=0,
    alpha_mse=1.0,
    beta_stft=0.2,
):
    combined_fn, _, _ = combined_masked_loss_terms(
        receptive_field_size,
        filter_column_idx=filter_column_idx,
        alpha_mse=alpha_mse,
        beta_stft=beta_stft,
    )
    return combined_fn


def combined_masked_loss_terms(
    receptive_field_size,
    filter_column_idx=0,
    alpha_mse=1.0,
    beta_stft=0.2,
):
    mse_fn = masked_mse(receptive_field_size, filter_column_idx)

    # Make STFT term spectral-only to avoid counting time MSE twice.
    stft_fn = masked_multires_stft_loss(
        receptive_field_size,
        filter_column_idx=filter_column_idx,
        w_time=0.0,
        w_mag=0.8,
        w_sc=0.2,
    )

    def loss_fn(y_true, y_pred):
        return alpha_mse * mse_fn(y_true, y_pred) + beta_stft * stft_fn(y_true, y_pred)

    def mse_component(y_true, y_pred):
        return mse_fn(y_true, y_pred)

    def stft_component(y_true, y_pred):
        return stft_fn(y_true, y_pred)

    # Stable metric names for TensorBoard/Keras logs.
    loss_fn.__name__ = "combined_masked_loss"
    mse_component.__name__ = "masked_mse"
    stft_component.__name__ = "masked_stft"

    return loss_fn, mse_component, stft_component
