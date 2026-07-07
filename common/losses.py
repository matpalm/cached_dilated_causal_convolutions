import tensorflow as tf


def masked_huber(receptive_field_size: int = None, alpha=0.1, reduce_mean: bool = True):
    """
    Calculates masked version of the Huber loss

    Parameters:
        receptive_field_size: number of initial time steps to ignore
        alpha: Huber delta; threshold when the loss transitions from quadratic to linear
    Returns:
        keras loss function
    """

    # TODO: generalise with MSE

    def loss_fn(y_true, y_pred):
        assert y_true.shape == y_pred.shape
        assert len(y_true.shape) == 3, "expected (batch, sequence_length, output_dim)"
        assert y_true.shape[-1] == 1, "expected (batch, sequence_length, output_dim=1)"
        if receptive_field_size:
            assert (
                y_true.shape[-2] > receptive_field_size
            ), "sequence is shorter than receptive_field_size!"
        # huber loss per element
        error = y_true - y_pred
        abs_error = tf.abs(error)
        quadratic = tf.minimum(abs_error, alpha)
        linear = abs_error - quadratic
        huber = 0.5 * tf.square(quadratic) + alpha * linear
        # average over elements of y
        huber = tf.reduce_mean(huber, axis=-1)
        if receptive_field_size:
            # we want to ignore the first elements of the loss since they
            # have been fed with left padded data
            huber = huber[:, receptive_field_size:]
        # return per-example average over sequence, optionally reduced over batch
        huber = tf.reduce_mean(huber, axis=-1)
        return tf.reduce_mean(huber) if reduce_mean else huber

    return loss_fn


def masked_mse(receptive_field_size: int = None, reduce_mean: bool = True):
    """
    Calculates masked version of mean square error

    Parameters:
        receptive_field_size: number of initial time steps to ignore
    Returns:
        keras loss function
    """

    def loss_fn(y_true, y_pred):
        assert y_true.shape == y_pred.shape
        assert len(y_true.shape) == 3, "expected (batch, sequence_length, output_dim)"
        assert y_true.shape[-1] == 1, "expected (batch, sequence_length, output_dim=1)"
        if receptive_field_size:
            assert (
                y_true.shape[-2] > receptive_field_size
            ), "sequence is shorter than receptive_field_size!"
        # average over elements of y
        mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
        # we want to ignore the first elements of the loss since they
        # have been fed with left padded data
        if receptive_field_size:
            mse = mse[:, receptive_field_size:]
        # return per-example average over sequence, optionally reduced over batch
        mse = tf.reduce_mean(mse, axis=-1)
        return tf.reduce_mean(mse) if reduce_mean else mse

    return loss_fn


def masked_multires_stft_loss(
    receptive_field_size: int = None,
    fft_sizes=(512, 1024, 2048),
    hop_sizes=(128, 256, 512),
    win_lengths=(512, 1024, 2048),
    w_time=0.5,
    w_mag=0.4,
    w_sc=0.1,
    reduce_mean: bool = True,
):
    """
    Calculates masked multi-resolution STFT loss

    Args:
        receptive_field_size: number of initial time steps to ignore
        fft_sizes: FFT sizes used at each STFT res
        hop_sizes: STFT hop sizes for each res
        win_lengths: STFT window lengths for each res
        w_time: Weight for time-domain MSE term
        w_mag: Weight for log-magnitude spectral term
        w_sc: Weight for spectral-convergence term
    """

    assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)

    def _stft_mag(x, fft_size, hop, win):
        s = tf.signal.stft(
            x,
            frame_length=win,
            frame_step=hop,
            fft_length=fft_size,
            window_fn=tf.signal.hann_window,
            # pad short tails (and very short sequences) to avoid empty STFT outputs
            pad_end=True,
        )
        return tf.abs(s)

    def loss_fn(y_true, y_pred):
        assert y_true.shape == y_pred.shape
        assert len(y_true.shape) == 3, "expected (batch, sequence_length, output_dim)"
        assert y_true.shape[-1] == 1, "expected (batch, sequence_length, output_dim=1)"
        if receptive_field_size:
            assert (
                y_true.shape[-2] > receptive_field_size
            ), "sequence is shorter than receptive_field_size!"

        # y: (batch, seq, channels=1)
        y_true_ = y_true
        y_pred_ = y_pred

        # mask left-padded receptive field
        if receptive_field_size and receptive_field_size > 0:
            y_true_ = y_true_[:, receptive_field_size:, :]
            y_pred_ = y_pred_[:, receptive_field_size:, :]

        # collapse channel dim for STFT (assuming 1 selected channel)
        y_true_1d = tf.squeeze(y_true_, axis=-1)
        y_pred_1d = tf.squeeze(y_pred_, axis=-1)

        time_mse = tf.reduce_mean(tf.square(y_true_1d - y_pred_1d), axis=-1)

        mr_mag = tf.constant(0.0, dtype=tf.float32)
        mr_sc = tf.constant(0.0, dtype=tf.float32)
        eps = tf.constant(1e-6, dtype=tf.float32)
        n = tf.constant(float(len(fft_sizes)), dtype=tf.float32)

        for fft_size, hop, win in zip(fft_sizes, hop_sizes, win_lengths):
            m_true = _stft_mag(y_true_1d, fft_size, hop, win)
            m_pred = _stft_mag(y_pred_1d, fft_size, hop, win)

            # log-mag L1 is usually more perceptual than linear-mag MSE
            log_true = tf.math.log(m_true + eps)
            log_pred = tf.math.log(m_pred + eps)
            mr_mag += tf.reduce_mean(tf.abs(log_true - log_pred), axis=[-2, -1])

            # spectral convergence
            num = tf.norm(m_true - m_pred, ord="euclidean", axis=[-2, -1])
            den = tf.norm(m_true, ord="euclidean", axis=[-2, -1]) + eps
            mr_sc += tf.math.divide_no_nan(num, den)

        mr_mag = tf.math.divide_no_nan(mr_mag, n)
        mr_sc = tf.math.divide_no_nan(mr_sc, n)

        total = w_time * time_mse + w_mag * mr_mag + w_sc * mr_sc
        total = tf.where(tf.math.is_finite(total), total, tf.zeros_like(total))
        return tf.reduce_mean(total) if reduce_mean else total

    return loss_fn


def combined_masked_loss(
    receptive_field_size: int = None,
    use_huber_loss: bool = False,
    alpha_mse: float = 1.0,
    beta_stft: float = 0.2,
    reduce_mean: bool = True,
):
    combined_fn, _, _ = combined_masked_loss_terms(
        receptive_field_size,
        use_huber_loss=use_huber_loss,
        alpha_mse=alpha_mse,
        beta_stft=beta_stft,
        reduce_mean=reduce_mean,
    )
    return combined_fn


def combined_masked_loss_terms(
    receptive_field_size: int = None,
    use_huber_loss: bool = False,
    alpha_mse: float = 1.0,
    beta_stft: float = 0.2,
    reduce_mean: bool = True,
):
    if use_huber_loss:
        core_loss_fn = masked_huber(receptive_field_size, reduce_mean=reduce_mean)
    else:
        core_loss_fn = masked_mse(receptive_field_size, reduce_mean=reduce_mean)

    # actually, STFT term can be spectral-only to avoid counting time MSE twice ?
    # so can remove w_time completely (?)
    stft_fn = masked_multires_stft_loss(
        receptive_field_size,
        w_time=0.0,
        w_mag=0.8,
        w_sc=0.2,
        reduce_mean=reduce_mean,
    )

    @tf.function
    def loss_fn(y_true, y_pred):
        return alpha_mse * core_loss_fn(y_true, y_pred) + beta_stft * stft_fn(
            y_true, y_pred
        )

    @tf.function
    def core_component(y_true, y_pred):
        return core_loss_fn(y_true, y_pred)

    @tf.function
    def stft_component(y_true, y_pred):
        return stft_fn(y_true, y_pred)

    # fix named metric names for tb / keras etc
    loss_fn.__name__ = "combined_masked_loss"
    core_component.__name__ = "masked_huber" if use_huber_loss else "masked_mse"
    stft_component.__name__ = "masked_stft"

    return (
        loss_fn,
        core_component,
        stft_component,
    )
