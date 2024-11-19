"""
Hyperparametrs and sweep configuration

Created on Sun nov 23 11:17:09 2022

@author: mukul
"""

from GlobalValues import BATCH_SIZE, EPOCHS, LEARNING_RATE, BETA, LOSS_FUNCTION
from LossFunctions import GMSE, LRMSE, jaccard_loss, two_branch_loss, GLMSE 


selected_case = 1

def get_HyperParams(selected_case):
    loss_cases = [
    # case 1 : GMSE
     {
        LEARNING_RATE: 3e-5,
        EPOCHS: 150,
        BATCH_SIZE: 16,
        LOSS_FUNCTION: GMSE
    },
    # case 2 : GLMSE global and local rmse
    {
        LEARNING_RATE: 3e-5,
        EPOCHS: 150,
        BATCH_SIZE: 16,
        LOSS_FUNCTION: GLMSE,
        BETA: 0.1
        #     beta = W_local_rmse / (w_local_rmse + w_global_rmse)
    },
    # case 3: jaccard_loss
     {
        LEARNING_RATE: 3e-5,
        EPOCHS: 150,
        BATCH_SIZE: 16,
        LOSS_FUNCTION: jaccard_loss
    },
    # case 4: two_branch_loss
     {
        LEARNING_RATE: 3e-5,
        EPOCHS: 150,
        BATCH_SIZE: 16,
        LOSS_FUNCTION: two_branch_loss,
        BETA: 0.81
        #     beta = W_rmse / (w_rmse + w_jaccard)
    },
    # case 5: local
     {
        LEARNING_RATE: 3e-5,
        EPOCHS: 150,
        BATCH_SIZE: 16,
        LOSS_FUNCTION: LRMSE
    }
     
]
    
    return loss_cases[selected_case - 1]


def get_SweepParams(selected_case):
    
    sweep_configuration_IOU_LRMSE = {
        'method': 'random',
        'name': 'AttentionUNET4Reg',
        # 'loss' : GMSE,
        'metric': {'goal': 'minimize', 'name': 'val_loss'},
        'parameters':
            {
                BATCH_SIZE: {'values': [16,32,64]},
                EPOCHS: {'values': [200]},
                LEARNING_RATE: {'max': 0.00009, 'min': 0.00001},
                BETA: {'values': [0.8]}
            }
    }
    
    cases = [
    # case 1 : GMSE
     {
        "SWEEP_OPERATION" : 1,
        LOSS_FUNCTION : GMSE,
        'sweep_configuration' : sweep_configuration_IOU_LRMSE
    }
    ]
    
    return cases[selected_case - 1]


# set the parmeter to run case with different loss function and hyperparameters for training / evaluation
use_config = get_HyperParams(4)


# set the parmeter to run case with different loss function and hyperparameters for real time application/blind testinf
real_time_config = {
        LEARNING_RATE: 3e-5,
        EPOCHS: 150,
        BATCH_SIZE: 16,
        LOSS_FUNCTION: GMSE
    }



