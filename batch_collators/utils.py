def norm_img(data):
    return data / 255.0


def img_channels_first(data):
    return data.permute(0, 3, 1, 2).contiguous()
