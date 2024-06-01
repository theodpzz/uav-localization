"""
Main script to train the AutoEncoder

Author: DI PIAZZA Theo
"""
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

#from src.model import AutoEncoder
#from src.dataset import DatasetUAV
#from src.utils import get_localization_accuracy

def validate(model, valid_loader, device, criterion):

  # train step
  model.eval()

  loss_validation = 0

  # iterate over dataloader
  for i, data in enumerate(valid_loader):

    # get data
    edges, _ = data
    edges = edges.to(device)

    # forward pass
    reconstructions = model(edges)

    # loss
    loss = criterion(reconstructions, edges.unsqueeze(1))

    loss_validation += loss.mean().item()

  return np.mean(loss_validation)

def train(model, train_loader, test_loader, device, optimizer, criterion, epochs=10):

  # train step
  model.train()

  loss_train, loss_test = [], []

  # iterate over epochs
  for epoch in range(epochs):

    loss_train_epoch = 0

    # iterate over dataloader
    for i, data in enumerate(train_loader):

      # get data
      edges, _ = data
      edges = edges.to(device)

      # forward pass
      reconstructions = model(edges)

      # loss
      loss = criterion(reconstructions, edges.unsqueeze(1))

      loss_train_epoch += loss.mean().item()

      # backward pass
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    # validation step
    loss_test_epoch = validate(model, test_loader, device, criterion)

    # save logs
    loss_train.append(loss_train_epoch)
    loss_test.append(loss_test_epoch)
    print(f'[EPOCH] {epoch} - [Losses] : train: {loss_train_epoch:.2f} | validation: {loss_test_epoch:.2f}')

  # compute localization accuracy
  accuracy, accuracy_kept = get_localization_accuracy(model, train_loader, test_loader, device)

  return accuracy, accuracy_kept

def main():

  # config
  path_images_train = 'example-data/train'
  path_images_test  = 'example-data/test'
  path_labels_train = 'example-data/train.csv'
  path_labels_test  = 'example-data/test.csv'
  
  batch_size = 2

  # device
  device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

  # data
  train_dataset = DatasetUAV(path_images_train, path_labels_train)
  train_loader  = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
  test_dataset  = DatasetUAV(path_images_test, path_labels_test)
  test_loader   = DataLoader(dataset = test_dataset, batch_size = 1, shuffle = False)

  # load model
  model = AutoEncoder()
  model = model.to(device)

  # optimizer, criterion
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  criterion = nn.MSELoss()

  # train and validate the model
  accuracy, accuracy_kept = train(model, train_loader, test_loader, device, optimizer, criterion)
  print(f'\nLocalization Accuracy: {accuracy:.2f} - Accuracy after Lowe ratio filter: {accuracy_kept:.2f} !')
