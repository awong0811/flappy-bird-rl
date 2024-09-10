DATA = {
    'resolution': 84,
    'batch_size': 32,
}

''' Example config.py
# Model-related configurations
MODEL = {
    'architecture': 'resnet50',  # Example model architecture
    'num_classes': 10,  # Number of classes for classification
    'pretrained': True,  # Whether to use a pretrained model
}

# Training-related configurations
TRAINING = {
    'epochs': 25,
    'learning_rate': 0.001,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'optimizer': 'Adam',  # Options could be 'SGD', 'Adam', etc.
    'scheduler': 'StepLR',  # Learning rate scheduler, e.g., 'StepLR', 'ReduceLROnPlateau'
    'scheduler_step_size': 7,
    'scheduler_gamma': 0.1,
}

# Logging and checkpoints
LOGGING = {
    'log_interval': 10,  # How often to log training progress
    'save_checkpoint': True,
    'checkpoint_path': './checkpoints',
    'checkpoint_interval': 5,  # Save checkpoint every X epochs
}

# Miscellaneous settings
MISC = {
    'seed': 42,  # Random seed for reproducibility
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # 'cuda' or 'cpu'
    'print_training_status': True,  # Print training status during training
}
'''