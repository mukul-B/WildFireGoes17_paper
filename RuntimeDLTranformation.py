
"""
This script contains 
1) Load Model from config file created during training
2) Transform Each image based on the Model selected


Created on Sun Jul 26 11:17:09 2020

@modified by: mukul badhan
on Sun Jul 23 11:17:09 2022

"""

import numpy as np
import torch

from GlobalValues import COLOR_NORMAL_VALUE, EPOCHS, BATCH_SIZE, LEARNING_RATE, LOSS_FUNCTION, RES_DECODER_PTH, RES_ENCODER_PTH, GOES_Bands, model_path
from GlobalValues import project_name_template,realtime_model_specific_postfix, RES_AUTOENCODER_PTH
from Autoencoder import Autoencoder as Selected_model
from torch.nn import Sigmoid 
from scipy.ndimage import gaussian_filter

class RuntimeDLTransformation:
    def __init__(self,conf):
        loss_function = conf.get(LOSS_FUNCTION)
        loss_function_name = str(loss_function).split("'")[1].split(".")[1]
        
        self.LOSS_NAME = loss_function_name
        OUTPUT_ACTIVATION = loss_function(1).last_activation
        self.selected_model = Selected_model(in_channels = GOES_Bands, out_channels = 1, last_activation = OUTPUT_ACTIVATION)
        model_name = type(self.selected_model).__name__

        project_name = project_name_template.format(
        model_name = model_name,
        loss_function_name=loss_function_name,
        n_epochs=conf.get(EPOCHS),
        batch_size=conf.get(BATCH_SIZE),
        learning_rate=conf.get(LEARNING_RATE),
        model_specific_postfix=realtime_model_specific_postfix
    )   
        print(project_name)
        path = model_path + project_name
        get_selected_model_weight(self.selected_model,path)
        self.selected_model.cuda()

    def Transform(self, x):
        
        x = single_dataload(x)
        with torch.no_grad():
            x = x.cuda()
            decoder_output = self.selected_model(x)
            if len(decoder_output) == 1:
                output_rmse, output_jaccard = None, None
                if self.LOSS_NAME == 'jaccard_loss':
                    output_jaccard = decoder_output
                else:
                    output_rmse = decoder_output
            else:
                output_rmse = decoder_output[0]
            return output_rmse
            
            
    def out_put_to_numpy(self, output_rmse):
        output_rmse = output_rmse.cpu()
        output = np.squeeze(output_rmse)

        return output
        


def single_dataload(x):
    x = np.array(x) / float(COLOR_NORMAL_VALUE)
    # x = np.expand_dims(x, 1)
    x = torch.Tensor(x)
    return x

def get_selected_model_weight(selected_model,model_project_path):
    # selected_model.load_state_dict(torch.load(model_project_path + "/" + RES_AUTOENCODER_PTH))
    selected_model.encoder.load_state_dict(torch.load(model_project_path + "/" + RES_ENCODER_PTH))
    selected_model.decoder.load_state_dict(torch.load(model_project_path + "/" + RES_DECODER_PTH))
    
    
