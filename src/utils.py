"""
Usefull functions to train the model
Author: DI PIAZZA Theo
"""
import math
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

def compute_lowe_and_error(train_loader, valid_loader, train_embeddings, train_names, embedding):
  """
  For a given embedding, computes similarity with all embeddings from train set
  Returns lowe's ratio and error (euclidean distance)
  """

  # correct format
  train_embeddings = np.array(train_embeddings)
  embedding = np.array([embedding.detach().cpu().tolist()[0]])

  # compute cosine similarity
  similarities = cosine_similarity(train_embeddings, embedding)

  # find 2 closest tiles
  indices = np.argsort(similarities.flatten())[-2:]
  values = similarities[indices]
  
  # compute lowe's ratio
  lowe_ratio = values[-1] /  values[-2]

  # compute euclidean error distance
  matching_tile_name = train_names[indices[-1]]

  # read coordinates
  x_match, y_match = train_loader.dataset.labels[train_loader.dataset.labels['name'] == matching_tile_name][['x', 'y']].values[0]
  x_val, y_val = valid_loader.dataset.labels[valid_loader.dataset.labels['name'] == validation_tile_name[0]][['x', 'y']].values[0]

  # compute euclidean distance
  error = math.sqrt((x_match - x_val)**2 + (y_match - y_val)**2)

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

def get_localization_accuracy(model, train_loader, valid_loader, error_threshold=50, lowe_threshold=1.13):
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
  for i, data in enumerate(valid_loader):
    edges, validation_tile_name = data
    edges = edges.to(device)

    # forward pass to get embeddings
    embeddings = model(edges, return_only_embedding=True)

    # compute lowe ratio and cosine similarity
    lowe_ratio, error = compute_lowe_and_error(dataloader, dataloader, train_embeddings, train_names, embeddings)

    lowe_ratios.append(lowe_ratio)
    errors.append(error)

  # compute metrics
  accuracy, accuracy_kept = compute_metrics(lowe_ratios, errors, error_threshold, lowe_threshold)

  return accuracy, accuracy_kept
