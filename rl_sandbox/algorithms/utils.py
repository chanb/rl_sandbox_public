def aug_data(data, num_aug, aug_batch_size):
        return data.repeat(1, num_aug, *[1] * (len(data.shape) - 2)).reshape(
            aug_batch_size, *data.shape[1:])
