from Models.SiameseModel import Siamese

class EmbeddingModule(Siamese):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dense(x)
        return x