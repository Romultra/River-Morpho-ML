# This module stores the functions needed for the training and inference steps of JamUNet

import torch

import torch.nn as nn
import numpy as np

from postprocessing.metrics import compute_metrics

# add the following code at the beginning of the notebook or .py file where the model is trained or tested. 
# if only one GPU is present you might need to remove the index "0" 
# torch.device('cuda:0') --> torch.device('cuda') / torch.cuda.get_device_name(0) --> torch.cuda.get_device_name() 

'''
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("CUDA Device Count: ", torch.cuda.device_count())
    print("CUDA Device Name: ", torch.cuda.get_device_name(0))
else:
    device = 'cpu'
    
print(f'Using device: {device}')
'''

def get_predictions(model, input_dataset, device='cuda:0'):
    '''
    Compute the predictions given the deep-learning model class and input dataset

    Inputs:
           model = class, deep-learning model
           input_dataset = TensorDataset or tensor, inputs for the model 
                           The input dataset has shape (n_samples, n_years_input=4, height=1000, width=500), 
           device = str, specifies the device where memory is allocated for performing the computations
                    default: 'cuda:0', other availble option: 'cpu'
                    Always remember to correctly set this key before running the simulations to avoid issues
    
    Output:
           predictions = tensor, model predictions with shape (B, H, W)
    '''    
    use_nb = device.startswith("cuda")
    # ensures tensor is on the right device; non_blocking only matters for CPU->GPU
    input_on_device = input_dataset.to(device, non_blocking=use_nb)
    predictions = model(input_on_device)
    return predictions.squeeze(1)


def training_unet(model, loader, optimizer, nonwater=0, water=1, pixel_size=60, water_threshold=0.5, device='cuda:0', 
                  loss_f='BCE', physics=False, alpha_er=1e-2, alpha_dep=1e-3, loss_er_dep='Huber', verbose: bool = False):

    '''
    Training loop for the deep-learning model. Allows to choose the loss function for binary classification.
    Enables the inclusion of physics-induced loss terms (regression losses).
    Inputs:
           model = class, deep-learning model
           loader = DataLoader, training data loader
           optimizer = torch.optim, optimizer for updating the model parameters
           nonwater = int, 'non-water' class pixel value
                      default: 0 (scaled classification). 
                      If the original classification is used, this key should be set to 1
           water = int, 'water' class pixel value
                   default: 1 (scaled classification). 
                   If the original classification is used, this should be set to 2
           pixel_size = int, image pixel resolution (m). Used for computing the erosion and deposition areas
                        default: 60, exported image resolution from Google Earth Engine
           water_threshold = float, threshold for classifying 'water' and 'non-water' pixels
                             default: 0.5, pixels with predicted probability >= threshold are classified as 'water'
           device = str, specifies the device where memory is allocated for performing the computations
                    default: 'cuda:0', other availble option: 'cpu'
                    Always remember to correctly set this key before running the simulations to avoid issues
           loss_f = str, binary classification loss function
                    default: 'BCE'. Other available options: 'BCE_Logits', 'Focal'
                    If other loss functions are set it raises an Exception
           physics = bool, enables the inclusion of physics-induced loss terms
                     default: False, physics-based losses are not included
           alpha_er = float, weight of the erosion loss term
                      default: 1e-2, adjust as needed when physics=True
           alpha_dep = float, weight of the deposition loss term
                       default: 1e-3, adjust as needed when physics=True
           loss_er_dep = str, regression loss function for erosion and deposition terms
                         default: 'Huber'. Other available options: 'RMSE', 'MAE'
                         If other loss functions are set it raises an Exception.
    
    Output:
           losses = list, training losses for each batch
    '''
    model.to(device)
    model.train()  # training mode

    use_nb = device.startswith("cuda")  # non_blocking only useful on CUDA
    losses = []

    num_batches = len(loader)
    # split in batches
    for batch_idx, batch in enumerate(loader):
        # move batch to device; non_blocking only matters CPU->GPU with pinned memory
        input = batch[0].to(device, non_blocking=use_nb)
        target = batch[1].to(device, non_blocking=use_nb)

        # get predictions
        predictions = get_predictions(model, input, device=device)
        
        # compute binary classification loss
        binary_loss = choose_loss(predictions, target, loss_f)
        
        # physics-induced loss terms
        if physics:
            # need binary predictions
            binary_predictions = (predictions >= water_threshold).float()

            # get real and predicted total areas of erosion and deposition
            real_erosion_deposition = get_erosion_deposition(
                input[0][-1], target, nonwater, water, pixel_size
            )
            predicted_erosion_deposition = get_erosion_deposition(
                input[0][-1], binary_predictions, nonwater, water, pixel_size
            )

            # compute physics-induced losses
            erosion_loss, deposition_loss = choose_er_dep_loss(
                real_erosion_deposition, predicted_erosion_deposition, loss_er_dep
            )

            # compute total loss
            loss = binary_loss + alpha_er * erosion_loss + alpha_dep * deposition_loss
        
        else:
            loss = binary_loss

        # update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose:
            # print every ~10% of an epoch, plus the last batch
            if num_batches <= 10 or batch_idx % max(1, num_batches // 10) == 0 or batch_idx == num_batches - 1:
                print(
                    f"  [train] batch {batch_idx + 1}/{num_batches}, "
                    f"loss={float(loss):.4f}"
                )

        # store loss as a Python scalar on CPU
        losses.append(loss.detach().cpu())

    return losses


def validation_unet(model, loader, nonwater=0, water=1, device='cuda:0', loss_f='BCE', water_threshold=0.5):
    '''
    Validation loop for the deep-learning model. Allows to choose the loss function for binary classification.
    Computes binary validation metrics by setting a threshold for water classification. 
    Physics-induced loss terms are not included.
    Inputs:
           model = class, deep-learning model
           loader = DataLoader, validation data loader
           nonwater = int, 'non-water' class pixel value
                      default: 0 (scaled classification). 
                      If the original classification is used, this key should be set to 1
           water = int, 'water' class pixel value
                   default: 1 (scaled classification). 
                   If the original classification is used, this should be set to 2
           device = str, specifies the device where memory is allocated for performing the computations
                    default: 'cuda:0', other availble option: 'cpu'
                    Always remember to correctly set this key before running the simulations to avoid issues
           loss_f = str, binary classification loss function
                    default: 'BCE'. Other available options: 'BCE_Logits', 'Focal'
                    If other loss functions are set it raises an Exception
           water_threshold = float, threshold for classifying 'water' and 'non-water' pixels
                             default: 0.5, pixels with predicted probability >= threshold are classified as 'water'
    Output:
           mean_loss = list, validation losses for each batch
           mean_acc = float, mean accuracy over all batches
           mean_prec = float, mean precision over all batches
           mean_rec = float, mean recall over all batches
           mean_f1 = float, mean F1-score over all batches
           mean_csi = float, mean Critical Success Index over all batches
    '''
    model.to(device)
    model.eval()  # evaluation mode

    use_nb = device.startswith("cuda")

    losses = []
    accuracies, precisions, recalls, f1_scores, csi_scores = [], [], [], [], []

    with torch.no_grad():
        for batch in loader:
            input = batch[0].to(device, non_blocking=use_nb)
            target = batch[1].to(device, non_blocking=use_nb)

            # get predictions
            predictions = get_predictions(model, input, device=device)

            # generate binary predictions
            binary_predictions = (predictions >= water_threshold).float()

            # compute loss 
            loss = choose_loss(predictions, target, loss_f)

            # compute metrics
            accuracy, precision, recall, f1_score, csi_score = compute_metrics(
                binary_predictions, target, nonwater, water
            )
            
            losses.append(loss.detach().cpu())
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1_score)
            csi_scores.append(csi_score)

    # aggregate metrics (assuming they are already Python floats or small tensors)
    mean_loss = losses  # you were returning the list; keep as-is to not break callers
    mean_acc = np.mean(accuracies)
    mean_prec = np.mean(precisions)
    mean_rec = np.mean(recalls)
    mean_f1 = np.mean(f1_scores)
    mean_csi = np.mean(csi_scores)

    return mean_loss, mean_acc, mean_prec, mean_rec, mean_f1, mean_csi


def choose_loss(preds, targets, loss_f='BCE'):
    '''
    Choose the binary classification loss function for the training and inference steps.
    Allows to choose among the following options: 
        - Binary Cross Entropy (BCE) loss, measures the difference between binary predictions 
          (by default not activated with a Sigmoid layer) and targets (which should be within the range [0,1]). [*]
        - BCE with Logits, combines the previous one with a Sigmoid layer 
          and it is said to be more stable than the single BCE activated by Sigmoid. [**]
        - Focal loss, updated BCE/BCE with Logits loss recommended for imbalanced datasets. 
          If a Sigmoid layer is not included in the network, the BCE with Logits adaptation should be used. [***]

    If "BCE_Logits" is chosen, the Sigmoid activation is implemented within the loss and should be removed in the network. 
    Despite being said to be more stable, this function generated instabilities during the training process.
    The use of the simple "BCE" function with a Sigmoid activation as final layer of the network is recommended. 
     
    Inputs: 
           preds = torch.tensor, predictions generated by the model
           targets = torch.tensor, targets of the dataset
           loss_f = str, binary classification loss function
                    default: 'BCE'. Other available options: 'BCE_Logits', 'Focal'
                    If other loss functions are set it raises an Exception

    Output: 
           loss = scalar, classification loss between predictions and targets 

    [*] from torch.nn.BCELoss (https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss)
    [**] from torch.nn.BCEWithLogitsLoss (https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss)
    [***] adapted from torch.nn.functional.binary_cross_entropy ()
          and torch.nn.functional.binary_cross_entropy_with_logits()
    '''      
    if loss_f == 'BCE':
        # requires sigmoid activation
        loss = nn.BCELoss()(preds, targets)
    elif loss_f == 'BCE_Logits':
        # sigmoid activated by default within the function
        loss = nn.BCEWithLogitsLoss()(preds, targets)
    elif loss_f == 'Focal':
        # allows to choose between adapted BCE and adapted BCE with Logits
        loss = FocalLoss()(preds, targets)

    else: 
        raise Exception('The specified loss function is wrong. Check the documentation for the available loss functions.')

    return loss

class FocalLoss(nn.Module):
    '''
    Focal loss for binary classification of the satellite images. Recommended for imbalanced datasets.
    '''
    
    def __init__(self, alpha=0.25, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        '''
        Inputs:
               alpha, gamma = float, hyperparameters of the loss function to be fine-tuned
               logits = bool, allows to choose between BCE and BCE with Logits (sigmoid activation within the loss)
                        default: False, loss function is BCE.
               reduce = bool, allows to get the mean value of the loss
                        default: False, full array is returned
        '''
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, preds, targets): 
        '''
        Inputs:
               preds = torch.tensor, predictions geenrated by the model
               targets = torch.tensor, targets of the dataset
        
        Output: 
               F_loss = Focal loss between predictions and targets 
        '''
        if self.logits:
            BCE_loss = nn.functional.binary_cross_entropy_with_logits(preds, targets, reduction='none')
        else:
            BCE_loss = nn.functional.binary_cross_entropy(preds, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def get_erosion_deposition(previous_year_img, current_year_img, nonwater=0, water=1, pixel_size=60):
    '''
    Compute the total areas of erosion and deposition. 
    Makes a pixel-wise comparison of the last input year and the target year image. 
    The area of erosion is the sum of 'water' pixels of the target year image 
    that were 'non-water' in the last input year image. 
    The area of deposition is the sum of 'non-water' pixels of the target year image
    that were 'water' in the last input year image.

    The total number of pixels of both the areas of erosion and deposition are multiplied 
    by the pixel area (the square of `pixel_size`).

    Inputs:
           previous_year_img = 2D array or tensor, representing previous year image
           current_year_img = 2D array or tensor, representing current year image
           nonwater = int, 'non-water' class pixel value
                      default: 0 (scaled classification). 
                      If the original classification is used, this key should be set to 1
           water = int, 'water' class pixel value
                   default: 1 (scaled classification). 
                   If the original classification is used, this should be set to 2
           pixel_size = int, image pixel resolution (m). Used for computing the erosion and deposition areas
                        default: 60, exported image resolution from Google Earth Engine
    
    Output:
           list, contains total areas of erosion and deposition in km^2
    '''  
    # sum erosion and deposition pixels
    erosion = torch.sum((previous_year_img == nonwater) & (current_year_img == water))
    deposition = torch.sum((previous_year_img == water) & (current_year_img == nonwater))
    
    # calculate areas of erosion and deposition
    factor = (pixel_size**2) / (1000**2) # conversion factor to get pixel area in km^2
    erosion_areas = erosion * factor
    deposition_areas = deposition * factor
    # .item() is required to correctly save the float number ### check
    return [erosion_areas.item(), deposition_areas.item()] 

def choose_er_dep_loss(preds, targets, loss_er_dep='Huber'):
    '''
    Choose the regression loss function of the total areas of erosion and deposition areas for the training step.

    It allows to choose among three options: 
        - Huber loss, computed with the pytorch function. It combines MAE and MSE loss, computed based 
          on the value of the error compared to the `delta` parameter (set to default = 1). [*]
        - Root Mean Square Error, computed by adding a square root to the pytorch Mean Square Error (MSE) function. [**]
        - Mean Absolut Error, computed using the pytorch function. [***]
    
    Given the intrinsic adaptability to the difference between target and prediction, the Huber loss is recommended. 
     
    Inputs: 
           preds = torch.tensor, predictions of total areas of erosion and deposition
           targets = torch.tensor, real areas of total erosion and deposition
           loss_er_dep = str, regression loss function for erosion and deposition terms
                         default: 'Huber'. Other available options: 'RMSE', 'MAE'
                         If other loss functions are set it raises an Exception.

    Output: 
           loss = scalar, regression loss with the specified function between predictions and targets 

    [*] from torch.nn.HuberLoss (https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html)
    [**] from torch.nn.MSELoss (https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html)
    [***] from torch.nn.L1Loss (https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html)
    '''
    if loss_er_dep == 'Huber':
        loss = nn.HuberLoss()(preds, targets)
    elif loss_er_dep == 'RMSE':
        loss = torch.sqrt(nn.MSELoss()(preds, targets))
    elif loss_er_dep == 'MAE':
        loss = nn.L1Loss()(preds, targets)
    else: 
        raise Exception('The specified loss function is wrong. Check the documentation for the available loss functions')
    
    return loss 