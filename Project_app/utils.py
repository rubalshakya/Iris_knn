import config
import pickle
import json
import numpy as np
import config

class SpeciesClass:
    
    def __init__(self,input_data):
        self.input_data = input_data


    def load_data(self):
        with open(config.model_path,"rb") as f:
            self.model = pickle.load(f)
        with open(config.project_data_path,"r") as f:
            self.project_data = json.load(f)
 
    def predict_species(self):
        self.load_data()
        test_array = []
        for feature in self.project_data["features"]:
            test_array.append(float(self.input_data[feature]))
 
        pred = self.model.predict([test_array])[0]

        for specie, encoded_val in self.project_data["label_encoded_data"].items():
            if pred == encoded_val:
                return specie
            
            

                

         