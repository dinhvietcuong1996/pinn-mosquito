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
# # Simulation Setup

# %%
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt


# Define the system of equations for numerical solution
def system(u, t, ):
    sigma, rho, beta = 10, 28, 8/3
    x, y, z = u
    return sigma * (y - x), x * (rho -  z) - y, x * y - beta * z

t_range = np.linspace(0, 20.0, 40*1000+1)

simulate_data = odeint(func=system, y0=[1.0, 1.0, 1.0], t=t_range)

# subplot for each variable
plt.figure(figsize=(10, 8))
plt.subplot(311)
plt.plot(t_range, simulate_data[:, 0])
plt.ylabel('x')
plt.subplot(312)
plt.plot(t_range, simulate_data[:, 1])
plt.ylabel('y')
plt.subplot(313)
plt.plot(t_range, simulate_data[:, 2])
plt.ylabel('z')
plt.show()

u_data = simulate_data

# %% [markdown]
# # PINN 1: original PINN framework




# %% [markdown]
# # PINN 1



# Define the system of equations for numerical solution
def system(u, t, ):
    sigma, rho, beta = 10, 28, 8/3
    x, y, z = u
    return sigma * (y - x), x * (rho -  z) - y, x * y - beta * z

t_range = np.linspace(0, 20.0, 40*100000+1)

simulate_data = odeint(func=system, y0=[1.0, 1.0, 1.0], t=t_range)

u_data = simulate_data
t_data = t_range

u_data = torch.tensor(u_data)
t_data = torch.tensor(t_data)



# %%
n_subdomains = 40
global_T_domain = [t_data.min(), t_data.max()]
subdomain_size = (global_T_domain[1] - global_T_domain[0]) / n_subdomains
overlap_size = 0.1 * subdomain_size
print(global_T_domain, subdomain_size, overlap_size)

t_init = torch.tensor([[0.0]])
u_init = torch.tensor([[1.0, 1.0, 1.0]])
MODELS = []
for i in range(n_subdomains):
    checkpoint_dir = f"checkpoints/PINN_Lorenz_forward_gradcausaldomain_subdomain_{i}"

    if checkpoint_dir is not None:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    T_training_subdomain = [global_T_domain[0] + i * subdomain_size - overlap_size, global_T_domain[0] + (i+1) * subdomain_size + overlap_size]
    T_training_subdomain[0] = max(T_training_subdomain[0], global_T_domain[0])
    T_training_subdomain[1] = min(T_training_subdomain[1], global_T_domain[1])
    T_prediction_subdomain = [global_T_domain[0] + i * subdomain_size, global_T_domain[0] + (i+1) * subdomain_size]
    T_prediction_subdomain[0] = max(T_prediction_subdomain[0], global_T_domain[0])
    T_prediction_subdomain[1] = min(T_prediction_subdomain[1], global_T_domain[1])

    t_data_subdomain = t_data[((t_data >= T_training_subdomain[0]) & (t_data <= T_training_subdomain[1])).reshape(-1)]
    u_data_subdomain = u_data[((t_data >= T_training_subdomain[0]) & (t_data <= T_training_subdomain[1])).reshape(-1)]
    U_subdomain = [u_data_subdomain.min(axis=0).values, u_data_subdomain.max(axis=0).values]
    U_subdomain_mean = (U_subdomain[0] + U_subdomain[1]) / 2
    U_subdomain[0] = torch.minimum(U_subdomain_mean - 1.0, U_subdomain[0])
    U_subdomain[1] = torch.maximum(U_subdomain_mean + 1.0, U_subdomain[1])

    print("Subdomain: ", i)
    print("Training domain: ", T_training_subdomain)
    print("Prediction domain: ", T_prediction_subdomain)

    lorenz_pinn = LorenzOdePINN(T_domain=T_training_subdomain,
                                    U_domain=U_subdomain,
                                    u_data=[t_init, u_init],
                                    lambda_alpha=0.99, 
                                    lambda_update=100, 
                                    n_gradual_steps=210000, 
                                    n_warmup_steps=10000, 
                                    checkpoint_dir=checkpoint_dir)
    
    if checkpoint_dir is None or not os.path.exists(os.path.join(checkpoint_dir, 'model_final.pt')):
        logs = lorenz_pinn.train(n_steps=300000,
                            n_epoches_per_evaluation=100,
                            n_patience=100,)
    else:
        lorenz_pinn.load_model(os.path.join(checkpoint_dir, 'model_final.pt'))
        MODELS.append([lorenz_pinn, T_prediction_subdomain])
    print("Overlap", T_prediction_subdomain[1] - overlap_size, T_training_subdomain[1])
    print()
    t_init = torch.linspace(T_prediction_subdomain[1] - overlap_size, T_training_subdomain[1], 100).view(-1, 1)
    u_init = lorenz_pinn.predict_u(t_init).detach()
    
