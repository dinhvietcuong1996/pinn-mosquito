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
from PINN import CausalTrainingGradientBalancingODePINN, OdePINN, MLP_UModel, ThetaModel, InverseMosquitoPINN
from PINNModels import MLP, LambdaLayer, ConstantVariable, LearnableVariable

# %% [markdown]
# # Simulation Setup

# %%
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

def get_params_culex(T=10, alpha="Petrovaradin"):
    if alpha == "Petrovaradin":
        alpha = 38
    elif alpha == "Bahariya":
        alpha = 35
    params = {
        "gamma_Aem": 1.143,
        "gamma_Ab": 0.885,
        "gamma_Ao": 2.0,
        "f_E": np.maximum(0, 0.16 * (np.exp(0.105 * (T-10)) - np.exp(0.105*(alpha-10) - (alpha-T)/(5.007)))),
        "f_P": np.maximum(0, 0.021 * (np.exp(0.162 * (T-10)) - np.exp(0.162*(alpha-10) - (alpha-T)/5.007))),
        "f_Ag": np.maximum(0, (T-9.8)/64.4),
        "mu_E": 0,
        "mu_L": 0.0304,
        "mu_P": 0.0146,
        "mu_em": 0.1,
        "mu_r": 0.08,
        "mu_A": 1/43,
        "kappa_L": 8e8,
        "kappa_P": 1e7,
        "sigma": 0.5,
        "beta_1": 141,
        "beta_2": 80,
    }
    params['m_A'] = np.maximum(-0.005941 + 0.002965 * T, params['mu_A'])
    params["f_L"] = params['f_P'] / 1.65
    params['m_E'] = params["mu_E"]
    params["m_L"] = params["mu_L"] + np.exp(-T/2)
    params["m_P"] = params["mu_P"] + np.exp(-T/2)
    return params

# Define the system of equations for numerical solution
def system(u, t, get_temperature, get_params):
    """
    inputs: a vector of  [E, L, P, Aem, Ab1, Ag1, Ao1, Ab2, Ag2, Ao2]
                          0  1  2  3    4    5    6    7    8    9
    """
    E, L, P, Aem, Ab1, Ag1, Ao1, Ab2, Ag2, Ao2 = u
    # get_temperature, get_params = args

    T = get_temperature(t)
    params = get_params(T)

    rescale_params = {'E': 1, 'L': 1, 'P': 1, 'Aem': 1, 'Ab1': 1, 'Ag1': 1, 'Ao1': 1, 'Ab2': 1, 'Ag2': 1, 'Ao2': 1}

    dE_dt = 1 / rescale_params['E'] * params['gamma_Ao'] * (params['beta_1'] * rescale_params['Ao1'] * Ao1 + params['beta_2'] * rescale_params['Ao2'] * Ao2) - (params['mu_E'] + params['f_E']) * E
    dL_dt = 1 / rescale_params['L'] * rescale_params['E'] * params['f_E'] * E - (params['m_L'] * (1 + L * rescale_params['L'] / params['kappa_L']) + params['f_L']) * L
    dP_dt = 1 / rescale_params['P'] * rescale_params['L'] * params['f_L'] * L - (params['m_P'] + params['f_P']) * P
    dAem_dt = 1 / rescale_params['Aem'] * params['f_P'] * rescale_params['P'] * P * params['sigma'] * np.exp(- params['mu_em'] * (1 + P * rescale_params['P'] / params['kappa_P'])) - (params['m_A'] + params['gamma_Aem']) * Aem
    dAb1_dt = 1 / rescale_params['Ab1'] * rescale_params['Aem'] * params['gamma_Aem'] * Aem - (params['m_A'] + params['mu_r'] + params['gamma_Ab']) * Ab1
    dAg1_dt = 1 / rescale_params['Ag1'] * rescale_params['Ab1'] * params['gamma_Ab'] * Ab1 - (params['m_A'] + params['f_Ag']) * Ag1
    dAo1_dt = 1 / rescale_params['Ao1'] * rescale_params['Ag1'] * params['f_Ag'] * Ag1 - (params['m_A'] + params['mu_r'] + params['gamma_Ao']) * Ao1
    dAb2_dt = 1 / rescale_params['Ab2'] * params['gamma_Ao'] * (rescale_params['Ao1'] * Ao1 + rescale_params['Ao2'] * Ao2) - (params['m_A'] + params['mu_r'] + params['gamma_Ab']) * Ab2
    dAg2_dt = 1 / rescale_params['Ag2'] * rescale_params['Ab2'] * params['gamma_Ab'] * Ab2 - (params['m_A'] + params['f_Ag']) * Ag2
    dAo2_dt = 1 / rescale_params['Ao2'] * rescale_params['Ag2'] * params['f_Ag'] * Ag2 - (params['m_A'] + params['mu_r'] + params['gamma_Ao']) * Ao2

    return [dE_dt, dL_dt, dP_dt, dAem_dt, dAb1_dt, dAg1_dt, dAo1_dt, dAb2_dt, dAg2_dt, dAo2_dt]


# %%
# Time range for the solution

def get_temperature(t):
    return 10 * np.sin(2 * np.pi / 365 * t) + 10
t_range = np.linspace(0, 365*3, 365*3*1000+1)

simulate_results = odeint(func=system, y0=np.ones((10,)) * 300, t=t_range, args=(get_temperature, get_params_culex,))


idx = np.isin(t_range, np.arange(365*2, 365*3))
t_data = torch.tensor(t_range[idx],).view(-1, 1)
u_data = torch.tensor(simulate_results[idx] )
a_data = get_temperature(t_data.detach().cpu().numpy())
a_data = torch.tensor(a_data)
params_data = get_params_culex(get_temperature(t_data.cpu().numpy()))
for key in params_data:
    params_data[key] = torch.tensor(params_data[key] * np.ones((t_data.shape[0], 1)))
assert t_data.shape[0] == np.arange(365*2, 365*3).shape[0]

print("u_data.shape = ", u_data.shape)



# %%
T_domain = [t_data.min().item(), t_data.max().item()]
U_domain = [u_data.min(axis=0).values, u_data.max(axis=0).values]
U_domain_mean = (U_domain[0] + U_domain[1]) / 2
U_domain[0] = torch.minimum(U_domain_mean - 1.0, U_domain[0])
U_domain[1] = torch.maximum(U_domain_mean + 1.0, U_domain[1])

Theta_domain = {}
for key in ['gamma_Aem', 'gamma_Ab', 'gamma_Ao', 'f_E', 'f_P', 'f_L', 'f_Ag', 'm_L', 'm_P', 'm_A']:
    Theta_domain[key] = [params_data[key].min(), params_data[key].max()]
    Theta_domain_mean = (Theta_domain[key][0] + Theta_domain[key][1]) / 2
    Theta_domain[key][0] = torch.minimum(Theta_domain_mean - 1.0, Theta_domain[key][0])
    Theta_domain[key][1] = torch.maximum(Theta_domain_mean + 1.0, Theta_domain[key][1])

print("T_domain", T_domain)
print("U_domain", U_domain)
print("Theta_domain", Theta_domain)
    
checkpoint_dir = "checkpoints/PINN_Mosquito_inverse"

if checkpoint_dir is not None:
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    final_checkpoint = os.path.join(checkpoint_dir, 'model_final.pt')

mosquito_pinn = InverseMosquitoPINN(T_domain=T_domain,
                            U_domain=U_domain,
                            Theta_domain=Theta_domain,
                            u_data=(t_data, u_data),
                            lambda_alpha=0.9, 
                            lambda_update=100, 
                            n_gradual_steps=0, 
                            n_warmup_steps=0, 
                            checkpoint_dir=checkpoint_dir)

if os.path.exists(final_checkpoint):
    mosquito_pinn.load_model(final_checkpoint)
else:
    logs = mosquito_pinn.train(n_steps=300000, 
                                n_epoches_per_evaluation=100,
                                n_patience=100,)
    



# %% [markdown]
# # END


