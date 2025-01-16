import os
from dotenv import load_dotenv

# Chargement des variables d'environnement depuis le fichier .env
load_dotenv()

# Variables de configuration
train_data_dir = os.getenv('TRAIN_DATA_DIR', 'path/to/train/data')
validation_data_dir = os.getenv('VALIDATION_DATA_DIR', 'path/to/validation/data')