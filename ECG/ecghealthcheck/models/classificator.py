import torch
import numpy as np
from PIL import Image
from typing import List
import matplotlib.pyplot as plt
from scipy.signal import resample
from sklearn.neighbors import KNeighborsClassifier
from ECG.ecghealthcheck.enums import ECGStatus
from ECG.ecghealthcheck.models.embedding import EmbeddingModel
from ECG.ecghealthcheck.models.gradcam import GradCAM


class Classificator():

    def __init__(self):

        extractor_params = {
            'kernel_size': 32,
            'num_features': 92,
            'activation_function': torch.nn.GELU,
            'normalization': torch.nn.BatchNorm1d,
            'dropout_rate': 0.2,
            'res_block_num': 4
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
            f='ECG/ecghealthcheck/networks/embedding_extractor_random.pth',
            map_location=self.device))
        self.embedding_extractor.train(False)

        self.classifier = KNeighborsClassifier(n_neighbors=3)

        self.abnorm_signal_for_xai = None

    def fit(self, norm_ecgs: List[torch.Tensor], abnorm_ecgs: List[torch.Tensor]):

        self.abnorm_signal_for_xai = abnorm_ecgs[0]

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

    def perform_xai(self, signal):

        cam = GradCAM(self.embedding_extractor)
        cam.register_hooks(self.embedding_extractor.res_blocks[-1])

        self.embedding_extractor.zero_grad()
        gr_cam_results = cam.compute_grads(signal, self.abnorm_signal_for_xai)

        ecgs = (
            signal[0].detach().cpu().numpy(),
            self.abnorm_signal_for_xai[0].detach().cpu().numpy()
        )

        fig, axs = plt.subplots(6, 2)
        fig.set_size_inches(18.5, 10.5)
        for i in range(len(ecgs)):

            for lead in range(6):
                
                data = ecgs[i][lead]

                heatmap = gr_cam_results[i][0].detach().numpy()
                heatmap = resample(heatmap, len(data))
                heatmap = (heatmap - np.min(heatmap)) / \
                    (np.max(heatmap) - np.min(heatmap))

                for j, heat_val in enumerate(heatmap):
                    axs[lead, i].axvline(x=j, color=(heat_val, 0, 0))
                axs[lead, i].plot(data)
                if lead == 0:
                    axs[lead, i].set_title('INPUT' if i == 0 else 'COMPARED', fontsize=20)

        fig.canvas.draw()
        
        plt.ioff()
        plt.close()

        rgb = fig.canvas.tostring_rgb()

        width, height = fig.canvas.get_width_height()

        img = Image.frombytes('RGB', (width, height), rgb)

        return img
