import sys
import os 
dir_path = os.path.dirname(__file__)[:os.path.dirname(__file__).rfind('\\')]
sys.path.append(dir_path)

import os
import random
import matplotlib.pyplot as plt
from Datasets.FewShotDataset import FewShotDataset
from Filtering.PreprocessingFilters import filter1
from Models.SiameseModel import Siamese
from Datasets.NoisyDataset import NoisyPairsDataset as NS_dataset
import torch.nn as nn
from Datasets.Chinese_PairsDataset import PairsDataset as DS_Chinese
from Datasets.NoisyDataset import NoisyPairsDataset as DS_Noisy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
from torch.utils.data import random_split


# Hyper params
#########################################################
SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
LR = 0.001
LOSS_FUNCTION = nn.BCELoss().cuda()
BATCH_SIZE = 16
WEIGHT_DECAY = 0.001
model = Siamese().to(DEVICE)
model.load_state_dict(torch.load('nets\SCNN.pth'))
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
THRESHHOLD = 0.5
SHOT = 10
dataset = FewShotDataset(shot=SHOT)
train_dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, generator=torch.Generator().manual_seed(SEED))
#########################################################

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


def show_history(history):

    plt.subplot(2,1,1)
    plt.plot(history['epochs'], history['train_accuracies'], label='Train accuracy')
    plt.plot(history['epochs'], history['test_accuracies'], label='Test accuracy')
    plt.ylim([0, 1.0])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(2,1,2)
    plt.plot(history['epochs'], history['train_losses'], label='Train loss')
    plt.plot(history['epochs'], history['test_losses'], label='Test loss')
    plt.ylim([0, 1.0])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.show()

history = {
    'epochs' : [0],
    'train_losses' : [1.],
    'test_losses' : [1.],
    'train_accuracies' : [0],
    'test_accuracies' : [0]
}
correct_preds = []


def train_epoch():

    steps_in_epoch = 0
    correct_predictions_in_epoch = 0
    epoch_loss = 0.0

    for TS_T, label in train_dl:

        steps_in_epoch += 1

        TS1, TS2, label = TS_T[0].to(DEVICE, non_blocking=True), TS_T[1].to(DEVICE, non_blocking=True), label.to(DEVICE, non_blocking=True)

        out = torch.reshape(model(TS1, TS2), (-1,))
        loss = LOSS_FUNCTION(out, label)

        epoch_loss += loss.item()
        correct_predictions_in_epoch += (torch.abs(out - label) < THRESHHOLD).count_nonzero().item()

        del out, loss

    return correct_predictions_in_epoch / len(dataset), epoch_loss / steps_in_epoch


def test_epoch(true_labels, ECGs):

    steps_in_epoch = 0
    correct_predictions_in_epoch = 0
    epoch_loss = 0.0

    for i in range(len(true_labels)):

        offset = int(i / SHOT)
        i_eq = np.random.randint(SHOT * offset, SHOT * (offset + 1))
        i_diff = np.random.randint(0, len(true_labels))
        while (i_diff >= SHOT * offset) and (i_diff <= SHOT * (offset + 1)):
            i_diff = np.random.randint(0, len(true_labels))


        ecg = ECGs[i][None, :, :].to(DEVICE)
        ecg_eq = ECGs[i_eq][None, :, :].to(DEVICE)
        ecg_diff = ECGs[i_diff][None, :, :].to(DEVICE)


        steps_in_epoch += 1

        label_true = torch.reshape(torch.as_tensor((1.), dtype=torch.float32), (-1,)).to(DEVICE)
        out = torch.reshape(model(ecg, ecg_eq), (-1,))
        loss = LOSS_FUNCTION(out, label_true)
        epoch_loss += loss.item()
        correct_predictions_in_epoch += (torch.abs(out - label_true) < THRESHHOLD).count_nonzero().item()
        del out, loss
        

        steps_in_epoch += 1

        label_false = torch.reshape(torch.as_tensor((0.), dtype=torch.float32), (-1,)).to(DEVICE)
        out = torch.reshape(model(ecg, ecg_diff), (-1,))
        loss = LOSS_FUNCTION(out, label_false)
        epoch_loss += loss.item()
        correct_predictions_in_epoch += (torch.abs(out - label_false) < THRESHHOLD).count_nonzero().item()
        del out, loss
    
    return correct_predictions_in_epoch / steps_in_epoch, epoch_loss / steps_in_epoch

if __name__ == '__main__':

    train_diagnoses, train_ECGs = dataset.get_train_data()
    test_diagnoses, test_ECGs = dataset.get_test_data()
    
    for epoch in range(EPOCHS):
        model.train(True)
        train_acc, train_loss = train_epoch()
        torch.cuda.empty_cache()

        model.train(False)
        test_acc, test_loss = test_epoch(test_diagnoses, test_ECGs)
        torch.cuda.empty_cache()

        history['epochs'].append(epoch + 1)
        history['train_losses'].append(train_loss)
        history['test_losses'].append(test_loss)
        history['train_accuracies'].append(train_acc)
        history['test_accuracies'].append(test_acc)

        print(f'Epoch: {epoch+1}\n\tTrain accuracy: {train_acc:.5f} -- Train loss: {train_loss:.5f}\n\tTest accuracy:  {test_acc:.5f} -- Test loss:  {test_loss:.5f}\n\n')

    if not os.path.exists('nets'):
        os.mkdir('nets')
    torch.save(model.state_dict(), 'nets\\FewShot_SCNN.pth')

    show_history(history)