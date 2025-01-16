from src.config import train_data_dir, validation_data_dir
from src.data_processing import get_data_generators
from src.model import create_model, train_model

def main():
    # Obtaining data generators
    train_generator, validation_generator = get_data_generators(train_data_dir, validation_data_dir)

    # Creating and training the model
    model = create_model()
    train_model(model, train_generator, validation_generator)

if __name__ == '__main__':
    main()