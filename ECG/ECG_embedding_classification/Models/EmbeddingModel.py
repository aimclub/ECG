from ECG.ECG_embedding_classification.Models.SiameseModel import Siamese


class EmbeddingModule(Siamese):
    def __init__(self,
                 kernel_size,
                 num_features,
                 like_LU_func,
                 norm1d,
                 dropout_rate
                 ):
        super(
            EmbeddingModule,
            self).__init__(
            kernel_size,
            num_features,
            like_LU_func,
            norm1d,
            dropout_rate)

    def forward(self, x):
        return self.forward_once(x)
