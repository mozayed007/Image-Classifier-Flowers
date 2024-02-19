# Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.models as models
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


# Model Loading Function implementation
def load_model(arch='efficientnet-b0', num_labels=102, hidden_units=512, class_to_idx=None):
    """
    Loads a pre-trained model for image classification.

    Args:
        arch (str): The architecture of the model. Options are 'vgg19', 'alexnet', and 'efficientnet-b0'.
        num_labels (int): The number of output labels for classification.
        hidden_units (int): The number of hidden units in the fully connected layers.

    Returns:
        torch.nn.Module: The loaded model.

    Raises:
        ValueError: If an unexpected network architecture is provided.
    """
    if arch == 'vgg19':
        model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        classifier_input_size = model.classifier[0].in_features
    elif arch == 'alexnet':
        model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        classifier_input_size = model.classifier[0].in_features
    elif arch == 'efficientnet-b0':
        model = EfficientNet.from_pretrained('efficientnet-b0')
        classifier_input_size = model._fc.in_features
    else:
        raise ValueError('Unexpected network architecture', arch)

    # Define the new classifier
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(classifier_input_size, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, num_labels)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    # Freeze all parameters in the model
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classifier in the model
    if arch in ['vgg19', 'alexnet']:
        model.classifier = classifier
    elif arch == 'efficientnet-b0':
        model._fc = classifier
    model.class_to_idx = class_to_idx
    # Unfreeze the classifier parameters
    for param in classifier.parameters():
        param.requires_grad = True

    return model

# Model Training Function implementation
def train_model(image_datasets, args ,arch='efficientnet-b0', hidden_units=512, epochs=25, learning_rate=0.001, gpu=True, checkpoint=None, class_to_idx=None):
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
        x: data.DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=4)
        for x in list(image_datasets.keys())
    }
    # Calculate dataset sizes.
    dataset_sizes = {
        x: len(dataloaders[x].dataset) 
        for x in list(image_datasets.keys())
    }    
    
    print('Network architecture:', arch)
    print('Number of hidden units:', hidden_units)
    print('Number of epochs:', epochs)
    print('Learning rate:', learning_rate)

    # Load the model     
    num_labels = len(image_datasets['train'].classes)
    model = load_model(arch=arch, num_labels=num_labels, hidden_units=hidden_units, class_to_idx=class_to_idx)
                
    # Defining criterion, optimizer and scheduler
    # Observe that only parameters that require gradients are optimized
    criterion =  nn.NLLLoss()

    optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, model.parameters())), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)    
    
    # Use gpu if selected and available
    if gpu and torch.cuda.is_available():
        print('Using GPU for training')
        device = torch.device("cuda:0")
        model = nn.DataParallel(model)  # Use data parallelism for training on multiple GPUs
        model.to(device)
        criterion.to(device)
    else:
        print('Using CPU for training')
        device = torch.device("cpu") 
        model.cpu()
        criterion.cpu()    
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

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

                    # Update progress bar description
                    t.set_description(f'Running {phase} Loss: {running_loss / ((t.n + 1) * inputs.size(0)):.4f} Running Accuracy: {running_corrects.double() / ((t.n + 1) * inputs.size(0)):.4f}')

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('Final {} Loss: {:.4f} Final Accuracy: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best validation Accuracy: {:4f}'.format(best_acc))

    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # # Store class_to_idx into a model property
    # model.class_to_idx = image_datasets['train'].class_to_idx
    
    # Save checkpoint if requested
    if checkpoint:
        print ('Saving checkpoint to:', checkpoint) 
        checkpoint_dict = {
            'arch': arch,
            'class_to_idx': model.class_to_idx, 
            'state_dict': model.state_dict(),
            'hidden_units': hidden_units ,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc
        }
        
        # Ensure the directory exists
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
    # Return the model
    return model

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
    args, _ = parser.parse_known_args()
    
    # Train model if invoked from command line
    if args.data_dir:    
        # Default transforms for the training, validation, and testing sets
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(45),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]),
            'valid': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        }
        
        # Load the datasets with ImageFolder
        image_datasets = {
            x: datasets.ImageFolder(root=args.data_dir + '/' + x, transform=data_transforms[x])
            for x in list(data_transforms.keys())
        }
        class_to_idx = image_datasets['train'].class_to_idx    
        train_model(image_datasets,args,class_to_idx) 
        
if __name__ == '__main__':
    main()