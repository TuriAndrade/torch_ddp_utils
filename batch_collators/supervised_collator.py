import torch


class SupervisedCollator(object):

    def __init__(
        self,
        data_transforms=[],
        label_transforms=[],
    ):
        super(SupervisedCollator, self).__init__()
        self.data_transforms = data_transforms
        self.label_transforms = label_transforms

    def __call__(self, batch):
        data, labels = zip(*batch)
        collated_data, collated_labels = torch.utils.data.default_collate(
            data,
        ), torch.utils.data.default_collate(
            labels,
        )

        for f in self.data_transforms:
            collated_data = f(collated_data)

        for f in self.label_transforms:
            collated_labels = f(collated_labels)

        return collated_data, collated_labels
