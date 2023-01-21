import sys
import json

class Config:
    def __init__(self):
        self.path_to_json = sys.modules[self.__module__].__name__+".json"
        with open(self.path_to_json, 'r') as f:
            self.config = json.load(f)

        self.device = self.config["device"]
        self.lr = self.config["lr"]
        self.batch_size = self.config["batch_size"]
        self.num_workers = self.config["num_workers"]
        self.img_size = self.config["img_size"]
        self.input_channels = self.config["input_channels"]
        self.l1_lambda = self.config["l1_lambda"]
        self.triplet_lambda = self.config["triplet_lambda"]
        self.lambda_gp = self.config["lambda_gp"]
        self.num_epochs = self.config["num_epochs"]
        self.load_model = self.config["load_model"]
        self.save_model = self.config["save_model"]
        self.save_period = self.config["save_period"]
        self.generator_model_path = self.config["generator_model_path"]

