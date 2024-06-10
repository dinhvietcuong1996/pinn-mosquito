# %%
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from scipy.integrate import odeint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import json
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
torch.set_default_device(device)
torch.set_default_dtype(torch.float64)
print("Setting default device to: ", device)
from PINN import CausalTrainingGradientBalancingODePINN, OdePINN, MLP_UModel, ThetaModel, LorenzThetaModel, LorenzOdePINN
from PINNModels import MLP, LambdaLayer, ConstantVariable, LearnableVariable


# %% [markdown]
# # PINN 1


# Define the system of equations for numerical solution
def system(u, t, ):
    sigma, rho, beta = 10, 28, 8/3
    x, y, z = u
    return sigma * (y - x), x * (rho -  z) - y, x * y - beta * z

t_range = np.linspace(0, 2.0, 40*1000+1)

simulate_data = odeint(func=system, y0=[1.0, 1.0, 1.0], t=t_range)

u_data = simulate_data

# %%
checkpoint_dir = "checkpoints/PINN_Lorenz_forward_gradcausal"

if checkpoint_dir is not None:
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    else:
        final_checkpoint = os.path.join(checkpoint_dir, 'model_final.pt')

U_domain = [torch.min(torch.tensor(u_data), axis=0).values, torch.max(torch.tensor(u_data), axis=0).values]
U_domain_mean = (U_domain[0] + U_domain[1]) / 2
U_domain[0] = torch.minimum(U_domain_mean - 1.0, U_domain[0])
U_domain[1] = torch.maximum(U_domain_mean + 1.0, U_domain[1])

lorenz_pinn = LorenzOdePINN(T_domain=[0.0, 2.0],
                                U_domain=U_domain,
                                u_data=[torch.tensor([[0.0]]), torch.tensor([[1.0, 1.0, 1.0]])],
                                lambda_alpha=0.99,
                                lambda_update=100, 
                                n_gradual_steps=200000, 
                                n_warmup_steps=1000, 
                                checkpoint_dir=checkpoint_dir)

if checkpoint_dir is None or not os.path.exists(os.path.join(checkpoint_dir, 'model_final.pt')):

    logs = lorenz_pinn.train(n_steps=300000,
                        n_epoches_per_evaluation=100,
                        n_patience=100,)
else:

    lorenz_pinn.load_model(os.path.join(checkpoint_dir, 'model_final.pt'))


# %%
lorenz_pinn.load_model(os.path.join(checkpoint_dir, 'model_final.pt'))


# %% [markdown]
# # END


