"""
Usefull functions to train the model

Author: DI PIAZZA Theo
"""
import math
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

def compute_lowe_and_error(train_loader, test_loader, train_embeddings, train_names, 
                           test_embedding, test_tile_name):
  """
  For a given embedding, computes similarity with all embeddings from train set
  Returns lowe's ratio and error (euclidean distance)
  """

  # correct format
  train_embeddings = np.array(train_embeddings)
  test_embedding   = np.array([test_embedding.detach().cpu().tolist()[0]])

  # compute cosine similarity
  similarities = cosine_similarity(train_embeddings, test_embedding)

  # find 2 closest tiles
  print(similarities.flatten(), type(similarities))
  indices = np.argsort(similarities.flatten())[-2:]
  print(f'similarities: {similarities}')
  values = similarities[indices]
  
  # compute lowe's ratio
  lowe_ratio = values[-1] /  values[-2]
  print(f'values: {values}')
  print(f'lowe_ratio: {lowe_ratio}')

  # compute euclidean error distance
  matching_tile_name = train_names[indices[-1]]

  # read coordinates
  x_match, y_match = train_loader.dataset.labels[train_loader.dataset.labels['name'] == matching_tile_name][['x', 'y']].values[0]
  x_test, y_test   = test_loader.dataset.labels[test_loader.dataset.labels['name'] == test_tile_name[0]][['x', 'y']].values[0]

  # compute euclidean distance
  error = math.sqrt((x_match - x_test)**2 + (y_match - y_test)**2)

  return lowe_ratio, error

def compute_metrics(lowe_ratios, errors, error_threshold, lowe_threshold):
  """
  Computes the localization accuracy given a error_threshold criterion
  Also computes the localizationa accuracy after applying filter with lowe's ratio
  """

  # localization accuracy
  accuracy = np.mean(1*(np.array(errors) <= error_threshold))

  # indices of prediction that are conserved with lowe ratio
  indices_to_keep = np.where(np.array(lowe_ratios) > lowe_threshold)[0]

  # compute accuracy based on conserved tiles
  if(indices_to_keep.shape[0] != 0):
    errors_kept     = np.array(errors)[indices_to_keep]
    accuracy_kept   = np.mean(1*(errors_kept <= error_threshold))
  else:
    accuracy_kept = 1.0

  return accuracy, accuracy_kept

def get_localization_accuracy(model, train_loader, test_loader, device, error_threshold=50, lowe_threshold=1.13):
  """
  From model and train, test dataloaders
  Returns the localization accuracy
  """

  # inference mode
  model.eval()

  # list that will stock train names, coordinates and embeddings
  train_names = []
  train_embeddings = []

  # lists that will stock lowe ratio and localization error
  lowe_ratios, errors = [], []

  # iterate over train dataloader
  for i, data in enumerate(train_loader):
    edges, name = data
    edges = edges.to(device)

    # forward pass to get embeddings
    embeddings = model(edges, return_only_embedding=True)

    # stock names, coordinates and embeddings
    train_names      += list(name)
    train_embeddings += embeddings.detach().cpu().tolist()

  # iterate over validation dataloader
  for i, data in enumerate(test_loader):
    edges, test_tile_name = data
    edges = edges.to(device)

    # forward pass to get embeddings
    test_embedding = model(edges, return_only_embedding=True)

    # compute lowe ratio and cosine similarity
    lowe_ratio, error = compute_lowe_and_error(train_loader, test_loader, train_embeddings, train_names, 
                                               test_embedding, test_tile_name)

    lowe_ratios.append(lowe_ratio)
    errors.append(error)

  # compute metrics
  accuracy, accuracy_kept = compute_metrics(lowe_ratios, errors, error_threshold, lowe_threshold)

  return accuracy, accuracy_kept
