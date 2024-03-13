# Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import glob
from torchvision.models import vgg19, alexnet, VGG19_Weights, AlexNet_Weights 
import time
import copy
import argparse
import os
from efficientnet_pytorch import EfficientNet
from collections  import OrderedDict
from tqdm import tqdm
import torch.distributed as dist

def unfreeze_layers(model, arch, num_layers_to_unfreeze):
    """
    Unfreezes the specified number of layers in the model based on the architecture.

    Args:
        model (torch.nn.Module): The model to unfreeze layers for.
        arch (str): The architecture of the model.
        num_layers_to_unfreeze (int): The number of layers to unfreeze.

    Returns:
        None
    """
    if arch in ['vgg19', 'alexnet']:
        # Get the number of features in the classifier
        num_features = len(list(model.features.children()))

        # Unfreeze the last num_layers_to_unfreeze layers
        for i, child in enumerate(model.features.children()):
            if i >= num_features - num_layers_to_unfreeze:
                for param in child.parameters():
                    param.requires_grad = True
    elif arch == 'efficientnet-b0':
        # Get the number of features in the _fc
        num_features = len(list(model._conv_stem.children()))

        # Unfreeze the last num_layers_to_unfreeze layers
        for i, child in enumerate(model._conv_stem.children()):
            if i >= num_features - num_layers_to_unfreeze:
                for param in child.parameters():
                    param.requires_grad = True

# Model Loading Function implementation
def load_model(arch='efficientnet-b0', num_labels=102, hidden_units=512, class_to_idx=None):
    """
    Loads a pre-trained model for image classification.

    Args:
        arch (str, optional): The architecture of the model. Options are 'vgg19', 'alexnet', and 'efficientnet-b0'. 
            Defaults to 'efficientnet-b0'.
        num_labels (int, optional): The number of output labels for classification. Defaults to 102.
        hidden_units (int, optional): The number of hidden units in the fully connected layers. Defaults to 512.
        class_to_idx (dict, optional): A dictionary mapping class names to class indices. Defaults to None.

    Returns:
        torch.nn.Module: The loaded model.

    Raises:
        ValueError: If an unexpected network architecture is provided.

    Examples:
        >>> model = load_model(arch='vgg19', num_labels=102, hidden_units=2048, class_to_idx=class_to_idx)
    """
    if arch == 'vgg19':
        model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        classifier_input_size = model.classifier[0].in_features
    elif arch == 'alexnet':
        model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        classifier_input_size = model.classifier[1].in_features
    elif arch == 'efficientnet-b0':
        model = EfficientNet.from_pretrained('efficientnet-b0')
        classifier_input_size = model._fc.in_features
    else:
        raise ValueError('Unexpected network architecture', arch)
    # Freeze all parameters in the model
    for param in model.parameters():
        param.requires_grad = False
    # Define the new classifier
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(classifier_input_size, (hidden_units // 2))),
        ('relu', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.2)),
        ('fc2', nn.Linear((hidden_units // 2), hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout2', nn.Dropout(p=0.2)),
        ('fc3', nn.Linear(hidden_units, num_labels)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    # Unfreeze the classifier parameters
    for param in classifier.parameters():
        param.requires_grad = True
    model.class_to_idx = class_to_idx
    # Replace the classifier in the model
    if arch in ['vgg19', 'alexnet']:
        model.classifier = classifier
    elif arch == 'efficientnet-b0':
        model._fc = classifier
    # Unfreeze the last layers
    unfreeze_layers(model, arch, 6)
    return model

# Model Training Function implementation
def train_model(image_datasets, args ,arch='efficientnet-b0', hidden_units=512, epochs=25, learning_rate=0.001, gpu=True, checkpoint=None, class_to_idx=None, device =torch.device("cuda:0")):
    """
    Trains a model using the provided image datasets and training parameters.

    Args:
        image_datasets (dict): A dictionary containing the image datasets for training and validation.
        args: Command line arguments.
        arch (str, optional): The architecture of the model. Defaults to 'efficientnet-b0'.
        hidden_units (int, optional): The number of hidden units in the model. Defaults to 512.
        epochs (int, optional): The number of epochs to train the model. Defaults to 25.
        learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.001.
        gpu (bool, optional): Whether to use GPU for training. Defaults to True.
        checkpoint (str, optional): The directory to save the checkpoint. Defaults to None.
        class_to_idx (dict, optional): A dictionary mapping class names to indices. Defaults to None.
        device (torch.device, optional): The device to use for training. Defaults to torch.device("cuda:0").

    Returns:
        model: The trained model.
    """
    # Use command line values when specified
    if args.arch:
        arch = args.arch     
        
    if args.hidden_units:
        hidden_units = args.hidden_units

    if args.epochs:
        epochs = args.epochs
            
    if args.learning_rate:
        learning_rate = args.learning_rate

    if args.gpu:
        gpu = args.gpu

    if args.save_dir:
        checkpoint = args.save_dir        
        
    # Using the image datasets, define the dataloaders
    dataloaders = {
        x: data.DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=8)
        for x in list(image_datasets.keys())
    }
    # Calculate dataset sizes.
    dataset_sizes = {
        x: len(dataloaders[x].dataset) 
        for x in list(image_datasets.keys())
    }    
    
    print(f'Network architecture: {arch}\n'
            f'Number of hidden units: {hidden_units}\n'
            f'Number of epochs: {epochs}\n'
            f'Learning rate: {learning_rate}\n'
            f'Using GPU: {gpu}\n'
            f'Save directory: {args.save_dir}')

    # Load the model     
    num_labels = len(image_datasets['train'].classes)
    model = load_model(arch=arch, num_labels=num_labels, hidden_units=hidden_units, class_to_idx=class_to_idx)

    # Defining criterion, optimizer and scheduler
    # Observe that only parameters that require gradients are optimized
    criterion =  nn.NLLLoss()
    optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, model.parameters())), lr=learning_rate)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                T_max = 32, # Maximum number of iterations.
                                eta_min = 1e-4) # Minimum learning rate.    
    
    # Use gpu if selected and available
    if gpu and device.type == 'cuda':
        print('Using GPU for training')
        model = nn.parallel.DistributedDataParallel(model)  # Use data parallelism for training on multiple GPUs
        model.to(device)
        criterion.to(device)
    else:
        print('Using CPU for training')
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    n_epochs_stop = 7  # Number of epochs to wait before stopping
    
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                
                model.train()  # Set model to training mode
                
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0  # Initialize total samples count
            # Iterate over data with progress bar.
            with tqdm(dataloaders[phase], unit="batch") as t:
                for inputs, labels in t:                
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            scheduler.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    total_samples += inputs.size(0)  # Update total samples count
                    # Update progress bar description
                    t.set_description(f'Running {phase} Loss: {running_loss / total_samples:.4f} Running Accuracy: {running_corrects.double() / total_samples:.4f}')

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('Final {} Loss: {:.4f} Final Accuracy: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            # # Print the labels
            # print("Labels:", labels)
            # # Print the predictions
            # print("Predictions:", preds)
            # # Compare labels and predictions
            # print("Correct predictions:", preds == labels.data)
            # At the end of each epoch, check validation loss
            if phase == 'valid':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if epoch_loss < best_val_loss:
                    # If the validation loss is lower than our current best, update the best loss and reset the count
                    best_val_loss = epoch_loss
                    epochs_no_improve = 0
                else:
                    # If the validation loss didn't improve, increment the count
                    epochs_no_improve += 1
                    if epochs_no_improve == n_epochs_stop:
                        print('Early stopping!')
                        # Load the best state dict
                        model.load_state_dict(best_model_wts)
                        # Exit the loop
                        break

        # If the model was early stopped, exit the outer loop as well
        if epochs_no_improve == n_epochs_stop:
            break

        print()

    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best validation Accuracy: {:4f}'.format(best_acc))
    # Define a default value for classifier
    classifier = None
    # Load best model weights
    model.load_state_dict(best_model_wts)
    # Check the architecture and set the classifier
    if arch in ['vgg19', 'alexnet']:
        classifier = model.module.classifier
        print(f'Model {arch} classifier saved.')
    elif arch == 'efficientnet-b0':
        classifier = model.module._fc
        print(f'Model {arch} classifier saved.')
    else:
        print(f"Architecture {args.arch} not recognized. Classifier not saved.")
    # Save checkpoint if requested
    if checkpoint:
        print ('Saving checkpoint to:', checkpoint) 
        checkpoint_dict = {
            'classifier': classifier,
            'arch': arch,
            'class_to_idx': model.module.class_to_idx, 
            'state_dict': model.state_dict(),
            'hidden_units': hidden_units ,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc
        }
        
        os.makedirs(checkpoint, exist_ok=True)
        # Get a list of existing checkpoint files for the current architecture
        existing_files = glob.glob(os.path.join(checkpoint, f'{arch}_model_checkpoint_*.pth'))
        # Determine the number for the new file
        if existing_files:
            # Get the highest existing number
            highest_num = max(int(file.split('_')[-1].split('.')[0]) for file in existing_files)
            new_num = highest_num + 1
        else:
            new_num = 0
        # Append a filename to the checkpoint directory
        checkpoint_file = os.path.join(checkpoint, f'{arch}_model_checkpoint_{new_num}.pth')
        
        torch.save(checkpoint_dict, checkpoint_file)
    return model

def load_data(data_dir):
    """
    Loads the image datasets from the specified directory and applies the necessary transformations.

    Args:
        data_dir (str): The directory path where the image datasets are located.

    Returns:
        dict: A dictionary containing the loaded image datasets for training, validation, and testing.
    """
    # Define your transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        x: datasets.ImageFolder(root=os.path.join(data_dir, x), transform=data_transforms[x])
        for x in ['train', 'valid', 'test']
    }

    return image_datasets


def main():
    # Define command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='Path to dataset ')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--arch', type=str, help='Model architecture')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, help='Number of hidden units')
    parser.add_argument('--save_dir', type=str, help='Save trained model checkpoint to file')
    args = parser.parse_args()
    # Determine the device to use
    if args.gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
        dist.init_process_group(backend='gloo', init_method='tcp://localhost:23456', rank=0, world_size=1)
    # Train model if invoked from command line
    if args.data_dir:   
        # load the datasets 
        image_datasets = load_data(args.data_dir)
        class_to_idx = image_datasets['train'].class_to_idx    
        train_model(image_datasets = image_datasets ,args = args ,class_to_idx = class_to_idx ,device = device) 
    
    
if __name__ == '__main__':
    main()