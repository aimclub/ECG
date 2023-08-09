from ECG.condition.models.siamese import SiameseModel


class EmbeddingModel(SiameseModel):
    def __init__(self,
                 kernel_size,
                 num_features,
                 like_LU_func,
                 norm1d,
                 dropout_rate
                 ):
        super(
            EmbeddingModel,
            self).__init__(
            kernel_size,
            num_features,
            like_LU_func,
            norm1d,
            dropout_rate)

    def forward(self, x):
        return self.forward_once(x)
