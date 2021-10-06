from utils.common import read_config
from utils.data_mgmt import get_data
from utils.model import create_model
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import argparse
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename= os.path.join(log_dir,"running_logs.log"),level=logging.INFO, format=logging_str, filemode="a")

def training(config_path, X_train, y_train, X_valid, y_valid):
    logging.info("Training starting...")
    config = read_config(config_path)

    LOSS_FUNCTION = config["params"]["loss_functions"]
    OPTIMIZER = config["params"]["optimizer"]
    METRICS = config["params"]["metrics"]
    NUM_CLASSES = config["params"]["num_classes"]

    model = create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES)

    EPOCHS = config["params"]["epochs"]
    VALIDATION_SET = (X_valid, y_valid)

    history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=VALIDATION_SET)
    logging.info("Training end")
    return model, history

def save_plot(plots_dir, history, file_name):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)

    os.makedirs(plots_dir, exist_ok=True)  
    plotPath = os.path.join(plots_dir, file_name)
    plt.savefig(plotPath)

    logging.info("Plot saved")


def test_model(model_clf, X_test, y_test):
    logging.info("Testing starting...")
    print("Model testing")
    print("-----"*10)
    model_clf.evaluate(X_test, y_test)
    logging.info("Testing end")

def save_model(model_dir, model):
    os.makedirs(model_dir, exist_ok=True)
    fileName = time.strftime("Model_%Y_%m_%d_%H_%M_%S_.h5")    
    model_path = os.path.join(model_dir, fileName)
    print(f"your model will be saved at the following location\n{model_path}")
    model.save(model_path)
    logging.info("Model saved")

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config.yaml")

    parsed_args = args.parse_args()

    config = read_config(parsed_args.config)
    validation_datasize = config["params"]["validation_datasize"]

    X_train, y_train, X_valid, y_valid, X_test, y_test = get_data(validation_datasize)

    model, history = training(parsed_args.config, X_train, y_train, X_valid, y_valid)

    plots_dir = config["artifacts"]["plots_dir"]
    models_dir = config["artifacts"]["models_dir"]

    save_plot(plots_dir, history, "plot-1")

    test_model(model, X_test, y_test)

    save_model(models_dir, model)

