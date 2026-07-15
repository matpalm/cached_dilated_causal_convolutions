# import tensorflow as tf


# def masked_mse(receptive_field_size, filter_column_idx=None):
#     """
#     Calculates masked version of mean square error

#     Parameters:
#         receptive_field_size: number of initial time steps to ignore
#         filter_column_idx: only calculate loss w.r.t this column in output. done since
#                            the output has 4 outs, but we might only care about one
#     Returns:
#         keras loss function
#     """

#     def loss_fn(y_true, y_pred):
#         assert len(y_true.shape) == 3, "expected (batch, sequence_length, output_dim)"
#         if filter_column_idx is not None:
#             # consider only a single column from output for loss
#             y_true = y_true[:, :, filter_column_idx : filter_column_idx + 1]
#             y_pred = y_pred[:, :, filter_column_idx : filter_column_idx + 1]
#         assert y_true.shape == y_pred.shape
#         # average over elements of y
#         mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
#         # we want to ignore the first elements of the loss since they
#         # have been fed with left padded data
#         mse = mse[:, receptive_field_size:]
#         # return average over batch and sequence
#         return tf.reduce_mean(mse)

#     return loss_fn
