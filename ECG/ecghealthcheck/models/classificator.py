import torch
from typing import List
from sklearn.neighbors import KNeighborsClassifier
from ECG.ecghealthcheck.enums import ECGStatus
from ECG.ecghealthcheck.models.embedding import EmbeddingModel


class Classificator():

    def __init__(self):

        extractor_params = {
            'kernel_size': 32,
            'num_features': 92,
            'activation_function': torch.nn.GELU,
            'normalization': torch.nn.BatchNorm1d,
            'dropout_rate': 0.2
        }

        self.embedding_extractor = EmbeddingModel(
            kernel_size=extractor_params['kernel_size'],
            num_features=extractor_params['num_features'],
            like_LU_func=extractor_params['activation_function'],
            norm1d=extractor_params['normalization'],
            dropout_rate=extractor_params['dropout_rate']
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_extractor.load_state_dict(torch.load(
            f='ECG/ecghealthcheck/networks/embedding_extractor.pth',
            map_location=self.device))
        self.embedding_extractor.train(False)

        self.classifier = KNeighborsClassifier(n_neighbors=3)

    def fit(self, norm_ecgs: List[torch.Tensor], abnorm_ecgs: List[torch.Tensor]):

        embeddings = []
        labels = []

        with torch.no_grad():
            for norm_ecg, abnorm_ecg in zip(norm_ecgs, abnorm_ecgs):

                embeddings.append(
                    torch.squeeze(self.embedding_extractor(norm_ecg)).detach().numpy()
                )
                labels.append(ECGStatus.NORM.value)

                embeddings.append(
                    torch.squeeze(self.embedding_extractor(abnorm_ecg)).detach().numpy()
                )
                labels.append(ECGStatus.ABNORM.value)

        self.classifier.fit(embeddings, labels)

    def predict(self, ecg: torch.Tensor) -> bool:
        with torch.no_grad():
            embedding = torch.squeeze(self.embedding_extractor(ecg)).detach().numpy()
            res = self.classifier.predict(embedding.reshape(1, -1))[0]
            return True if res == ECGStatus.NORM.value else False
