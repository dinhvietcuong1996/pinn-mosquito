
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import json
import os
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
torch.set_default_device(device)
torch.set_default_dtype(torch.float64)
print("Setting default device to: ", device)


from PINNModels import MLP, LearnableVariable, ConstantVariable, LambdaLayer




class OdePINN:

    def __init__(self, 
                 T_domain=[0, 1], 
                 U_domain=None, 
                 Theta_domain=None,
                 data_loss_weight=1.0, 
                 ode_loss_weight=1.0, 
                 **kwargs):
        
        self.T_domain = T_domain
        self.U_domain = U_domain
        self.Theta_domain = Theta_domain
        self._compile_normalization_fn()
        self.data_loss_weight = data_loss_weight
        self.ode_loss_weight = ode_loss_weight
        self.build_theta_model()
        self.build_u_model()
        self.build_optimizer()
        self.trainable_parameters = self._get_trainable_parameters()
        self.logs = {"train": [], "val": [],}
        self.early_stopping_step = 0

    def _compile_normalization_fn(self, **kwargs):
        if self.T_domain is None:
            raise ValueError("T_domain must be provided")
        else:
            self.input_norm_params = {"min": self.T_domain[0], "max": self.T_domain[1]}

        if self.U_domain is None:
            self.output_norm_params = None
        else:
            self.output_norm_params = {"min": self.U_domain[0], "max": self.U_domain[1]}

        if self.Theta_domain is None:
            self.theta_norm_params = None
        else:
            self.theta_norm_params = self.Theta_domain

    def build_theta_model(self, **kwargs):
        self.theta_model = ThetaModel(input_norm_params=self.input_norm_params,
                                      output_norm_params=self.theta_norm_params,
                                      **kwargs)
    
    def predict_theta(self, t):
        theta_values = self.theta_model.forward(t, unnormalize=True)
        return theta_values

    def build_u_model(self, **kwargs):
        self.u_model = MLP_UModel(input_size=1, 
                                output_size=10,
                                hidden_layer_sizes=[100, 100, 100, 100],
                                activation=nn.GELU(),
                                last_activation=nn.Identity(),
                                input_norm_params=self.input_norm_params,
                                output_norm_params=self.output_norm_params,)
    
    def predict_u(self, t):
        u = self.u_model.forward(t)
        u = self.u_model.output_unnorm_fn(u)
        return u

    def u_data_fn(self, **kwargs):
        # should return x, y
        raise NotImplementedError

    def ode_residual_fn(self, training=True, **kwargs):
        batch_size = 1000 if training else 10000
        if not training:
            return torch.linspace(self.input_norm_params['min'], self.input_norm_params['max'], steps=batch_size, requires_grad=True).reshape(-1, 1)
        return torch.rand(size=(batch_size, 1), requires_grad=True) * (self.input_norm_params['max'] - self.input_norm_params['min']) + self.input_norm_params['min']

    def u_data_loss_fn(self, data, **kwargs):
        x, y_true = data
        y_pred = self.u_model.forward(x)
        y_true = self.u_model.output_norm_fn(y_true)
        rmse = torch.mean((y_pred - y_true)**2)
        return [rmse]
    
    def ode_system(self, u_values, theta_values, **kwargs):
        raise NotImplementedError
    
    def ode_equations(self, data):
        inputs = data
        u_values = self.u_model.forward(inputs, unnormalize=False)
        theta_values = self.theta_model.forward(inputs, unnormalize=True)

        n_u_values = u_values.shape[1]
        list_gradient_normed_u = []
        for ii in range(n_u_values):
            normed_u = u_values[:, ii].unsqueeze(-1)
            gradient_normed_u = torch.autograd.grad(normed_u, inputs, 
                                                    grad_outputs=torch.ones_like(normed_u), 
                                                    create_graph=True)[0]
            list_gradient_normed_u.append(gradient_normed_u)

        u_unnormed_values = self.u_model.output_unnorm_fn(u_values)
        list_du_dt = self.ode_system(u_values=u_unnormed_values, 
                                     theta_values=theta_values)
        
        if self.U_domain is not None:
            list_du_dt = [du_dt * 2 / (self.u_model.output_norm_params['max'][ii] - self.u_model.output_norm_params['min'][ii]) 
                          for ii, du_dt in enumerate(list_du_dt)]
            
        list_equations = []
        for ii, gradient_normed_u in enumerate(list_gradient_normed_u):
            list_equations.append(gradient_normed_u - list_du_dt[ii])

        return list_equations    

    def ode_loss_fn(self, data, **kwargs):
        equations = self.ode_equations(data)
        return [torch.mean(equation**2) for equation in equations]
    
    def compute_total_loss(self, step=None, training=True):
        total_loss = torch.tensor(0.0)
        loss_terms = []

        # data
        data_ = self.u_data_fn(step=step, training=training)
        data_losses = self.u_data_loss_fn(data_, step=step, training=training)
        for loss in data_losses:
            total_loss += self.data_loss_weight * 1/len(data_losses) * loss
            loss_terms.append(loss)

        # ode
        ode_data = self.ode_residual_fn(step=step, training=training)
        ode_losses = self.ode_loss_fn(ode_data, step=step, training=training)
        for loss in ode_losses:
            total_loss += self.ode_loss_weight * 1/len(ode_losses) * loss
            loss_terms.append(loss)

        return total_loss, loss_terms

    
    def _get_trainable_parameters(self,):
        trainable_parameters = []
        trainable_parameters += list(self.u_model.parameters())
        if self.theta_model is not None:
            trainable_parameters += list(self.theta_model.parameters())
        trainable_parameters = [param for param in trainable_parameters if param.requires_grad]
        trainable_parameters = list(set(trainable_parameters))
        return trainable_parameters

    def build_optimizer(self, learning_rate=1e-3,):
        trainable_parameters = self._get_trainable_parameters()
        self.optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate)

    def train_step(self, step=None, **kwargs):
        self.optimizer.zero_grad()
        total_loss, loss_terms = self.compute_total_loss(step=step, training=True)
        total_loss.backward()
        self.optimizer.step()

    def get_current_weights(self, ):
        return {
            "u_model": self.u_model.get_weights(),
            "theta_model": self.theta_model.get_weights(),
        }
    
    def save_model(self, path):
        current_weights = self.get_current_weights()
        torch.save(current_weights, path)

    def load_model(self, path):
        weights = torch.load(path)
        self.load_weights(weights)

    def load_weights(self, weights):
        self.u_model.set_weights(weights["u_model"])
        self.theta_model.set_weights(weights["theta_model"])

    def early_stopping_callback(self, step=None, n_patience=100, **kwargs):
        if not hasattr(self, "best_loss"):
            self.best_loss = None
        if not hasattr(self, "patience"):
            self.patience = 0

        cur_loss = self.logs['val'][-1][1]
        if step < self.early_stopping_step:
            return
        if self.best_loss is None or cur_loss < self.best_loss:
            self.best_loss = cur_loss
            self.best_weights = self.get_current_weights()
            self.patience = 0
        else:
            self.patience += 1
            if self.patience > n_patience:
                self.load_weights(self.best_weights)
                self.is_training = False

    def verbose_callback(self, **kwargs):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        verbose_data = {"Time": current_time, 
            'Train Loss': self.logs["train"][-1][1], 
            'Val Loss': self.logs["val"][-1][1], 
            "Data loss": self.logs["val"][-1][2][0],
            "ODE Loss": sum(self.logs["val"][-1][2][1:]),
        }
        self.pbar.set_postfix(verbose_data)

    def custom_end_of_step_callback(self, **kwargs):
        pass

    def custom_end_of_evaluation_callback(self, step=None, **kwargs):
        pass

    def custom_end_of_training_callback(self, **kwargs):
        pass

    def train(self, n_steps=10000,
                    n_epoches_per_evaluation=100,
                    n_patience=100,
                    **kwargs):
        self.train_params = {
            "n_steps": n_steps,
            "n_epoches_per_evaluation": n_epoches_per_evaluation,
            "n_patience": n_patience,
        }
        for key in kwargs:
            self.train_params[key] = kwargs[key]
            
        self.is_training = True
        if n_patience is None:
            n_patience = n_steps
            
        self.pbar = tqdm.tqdm(range(0, n_steps), mininterval=1)
        for step in self.pbar:
            self.train_step(step=step)
            self.custom_end_of_step_callback(step=step)

            if step % n_epoches_per_evaluation == 0 or step == n_steps - 1:
                total_loss, loss_terms = self.compute_total_loss(step=step, training=True)
                self.logs['train'].append([step, total_loss.item(), [loss.item() for loss in loss_terms]])
                
                total_loss, loss_terms = self.compute_total_loss(step=step, training=False)
                self.logs['val'].append([step, total_loss.item(), [loss.item() for loss in loss_terms]])

                self.custom_end_of_evaluation_callback(step=step)
                self.early_stopping_callback(step=step, n_patience=n_patience)
                self.verbose_callback()
                if not self.is_training:
                    break
        if hasattr(self, "best_weights"):
            self.load_weights(self.best_weights)
        self.custom_end_of_training_callback()
        return self.logs
    
class CausalTrainingGradientBalancingODePINN(OdePINN):

    def __init__(self, 
                 u_data=None,
                 lambda_alpha=0.9, 
                 lambda_update=100, 
                 n_gradual_steps=10000, 
                 n_warmup_steps=1000, 
                 checkpoint_dir=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_gradual_steps = n_gradual_steps 
        self.n_warmup_steps = n_warmup_steps
        self.ode_loss_weight = 0.0
        self.lambda_update = lambda_update
        self.lambda_alpha = lambda_alpha
        self.checkpoint_dir = checkpoint_dir
        self.u_data = u_data

        ode_data = self.ode_residual_fn(step=None, training=False)
        ode_losses = self.ode_loss_fn(ode_data, step=None, training=False)
        self.n_ode_terms = len(ode_losses)
        self.ode_lambdas = [1.0 for _ in range(self.n_ode_terms)]
        self.logs['ode_lambdas'] = []
        self.lambda_update_mode = 1
        self.early_stopping_step = max(100000, self.n_gradual_steps)

    def u_data_fn(self, **kwargs):
        return self.u_data

    def compute_total_loss(self, step=None, training=True):
        total_loss = torch.tensor(0.0)
        loss_terms = []

        # data
        cur_data = self.u_data_fn(step=step, training=training)
        cur_loss = self.u_data_loss_fn(cur_data, step=step, training=training)
        data_loss = torch.concat([l.reshape(-1,) for l in cur_loss], dim=0)
        data_loss = torch.mean(data_loss)
        total_loss += 1/len(cur_loss) * self.data_loss_weight * data_loss
        loss_terms.append(data_loss)

        # ode
        ode_data = self.ode_residual_fn(step=step, training=training)
        ode_losses = self.ode_loss_fn(ode_data, step=step, training=training)
        for ii, loss in enumerate(ode_losses):
            lambda_ode = 1.0
            if training:
                lambda_ode = self.ode_lambdas[ii]
            total_loss += lambda_ode * self.ode_loss_weight * 1 / len(ode_losses) * loss
            loss_terms.append(loss)

        return total_loss, loss_terms

    def update_ode_lambdas(self, step=None, training=True, lambda_alpha=None, return_gradients=False, **kwargs):
        if lambda_alpha is None:
            lambda_alpha = self.lambda_alpha
        list_average_gradients = []
        # data
        cur_data = self.u_data_fn(step=step, training=training)
        cur_loss = self.u_data_loss_fn(cur_data, step=step, training=training)
        data_loss = torch.concat([l.reshape(-1,) for l in cur_loss], dim=0)
        data_loss = torch.mean(data_loss)
        data_gradients = torch.autograd.grad(data_loss, self.trainable_parameters, create_graph=True, allow_unused=True)
        if self.lambda_update_mode == 1:
            list_mean_gradients = [torch.mean(torch.abs(gradient)) for gradient in data_gradients if gradient is not None]
            mean_data_gradients = torch.mean(torch.stack(list_mean_gradients)).detach().item()
        elif self.lambda_update_mode == 2:
            list_mean_gradients = [torch.sqrt(torch.mean(torch.square(gradient))) for gradient in data_gradients if gradient is not None]
            mean_data_gradients = torch.mean(torch.stack(list_mean_gradients)).detach().item()

        if mean_data_gradients == 0:
            mean_data_gradients = 1
        list_average_gradients.append(mean_data_gradients)
        # ode
        ode_data = self.ode_residual_fn(step=step, training=training)
        ode_losses = self.ode_loss_fn(ode_data, step=step, training=training)
        for ii, loss in enumerate(ode_losses):
            if self.ode_loss_weight > 0.0:
                if self.lambda_update_mode == 1:
                    list_max_gradients = [torch.max(torch.abs(gradient)) for gradient in 
                                        torch.autograd.grad(loss, self.trainable_parameters, create_graph=True, allow_unused=True)
                                        if gradient is not None]
                    max_gradients = torch.max(torch.stack(list_max_gradients)).detach().item()
                elif self.lambda_update_mode == 2:
                    list_max_gradients = [torch.sqrt(torch.mean(torch.square(gradient))) for gradient in 
                                        torch.autograd.grad(loss, self.trainable_parameters, create_graph=True, allow_unused=True)
                                        if gradient is not None]
                    max_gradients = torch.mean(torch.stack(list_max_gradients)).detach().item()
                list_average_gradients.append(max_gradients)
                self.ode_lambdas[ii] = lambda_alpha * self.ode_lambdas[ii] + (1 - lambda_alpha) *  mean_data_gradients / max_gradients
        
        if return_gradients:
            return list_average_gradients
        return self.ode_lambdas
    
    def ode_residual_fn(self, step=0, training=True, **kwargs):
        batch_size = 1000 if training else 10000
        if not training:
            return torch.linspace(self.input_norm_params['min'], self.input_norm_params['max'], steps=batch_size, requires_grad=True).reshape(-1, 1)
        
        if step <= self.n_gradual_steps:
            data = torch.linspace(0, 1, steps=batch_size, requires_grad=True).reshape(-1, 1)
            data = data * max(0, (step - self.n_warmup_steps + 1) / (self.n_gradual_steps - self.n_warmup_steps + 1))
            data = data * (self.input_norm_params['max'] - self.input_norm_params['min']) + self.input_norm_params['min']
            return data
        
        return torch.rand(size=(batch_size, 1), requires_grad=True) * (self.input_norm_params['max'] - self.input_norm_params['min']) + self.input_norm_params['min']
    
    def custom_end_of_step_callback(self, step=0,):
        if step == self.n_warmup_steps:
            print("Setting ODE loss weight to 1.0")
            self.ode_loss_weight = 1.0
        elif step == self.n_gradual_steps:
            print("Done with gradual training.")

        if self.lambda_update is not None and step % self.lambda_update == 0 and self.ode_loss_weight > 0.0:
            self.update_ode_lambdas(step=step, training=True)

    def early_stopping_callback(self, step=None, n_patience=100, **kwargs):
        if step < self.n_gradual_steps:
            return
        super().early_stopping_callback(step=step, n_patience=n_patience, **kwargs)

    def verbose_callback(self, **kwargs):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        verbose_data = {"Time": current_time, 
            'Train Loss': self.logs["train"][-1][1], 
            'Val Loss': self.logs["val"][-1][1],
            "Data loss": self.logs["val"][-1][2][0],
            "ODE Loss": sum(self.logs["val"][-1][2][1:]),
            "Lambda": " ".join([f"{lam:.4e}" for lam in self.ode_lambdas]),
        }
        self.pbar.set_postfix(verbose_data)

    def custom_end_of_training_callback(self, **kwargs):
        super().custom_end_of_training_callback(**kwargs)
        if self.checkpoint_dir is not None:
            cur_model_filepath = os.path.join(self.checkpoint_dir, 'model_final.pt')
            self.save_model(cur_model_filepath)
            
            cur_log_filepath = os.path.join(self.checkpoint_dir, 'log.json')
            with open(cur_log_filepath, 'w') as f:
                json.dump(self.logs, f, indent=4)

    def custom_end_of_evaluation_callback(self, step=None, **kwargs):
        self.logs['ode_lambdas'].append(self.ode_lambdas.copy())
        super().custom_end_of_evaluation_callback(step=step, **kwargs)
        if self.checkpoint_dir is not None:
            cur_model_filepath = os.path.join(self.checkpoint_dir, 'model_{}.pt'.format(step))
            self.save_model(cur_model_filepath)


### U-Model

class UModel:

    def __init__(self, 
                 input_norm_params=None,
                 output_norm_params=None,
                 **kwargs):
        
        self.input_norm_params = input_norm_params
        self.output_norm_params = output_norm_params
        
        self.input_norm_fn, self.input_unnorm_fn = self._compile_normalization_fn(norm_params=input_norm_params)
        self.output_norm_fn, self.output_unnorm_fn = self._compile_normalization_fn(norm_params=output_norm_params)

    def _compile_normalization_fn(self, norm_params=None, **kwargs):
        if norm_params is None:
            return lambda x: x, lambda x: x
        else:
            # to [-1, 1]
            min_val = norm_params["min"]
            max_val = norm_params["max"]
            norm_fn = lambda x: (x - min_val) / (max_val - min_val) * 2 - 1
            unnorm_fn = lambda x: (x + 1) / 2 * (max_val - min_val) + min_val
            return norm_fn, unnorm_fn

    def forward(self, x, unnormalize=False):
        x = self.input_norm_fn(x)
        x = self.model.forward(x)
        if unnormalize:
            return self.output_unnorm_fn(x)
        return x
    
    def parameters(self,):
        return self.model.parameters()
    
    def get_weights(self,):
        weight_values = {}
        for key, weight in self.model.state_dict().items():
            weight_values[key] = weight.clone()
        return weight_values
    
    def set_weights(self, weight_values):
        self.model.load_state_dict(weight_values)

class MLP_UModel(UModel):

    def __init__(self, 
                 input_norm_params=None,
                 output_norm_params=None,
                 hidden_layer_sizes=(100, 100, 100), 
                 input_size=1, 
                 output_size=10, 
                 activation=nn.GELU(), 
                 last_activation=nn.Identity(), 
                 **kwargs):
        
        super().__init__(input_norm_params=input_norm_params,
                         output_norm_params=output_norm_params,)

        self.model = MLP(hidden_layer_sizes=hidden_layer_sizes,
                         input_size=input_size,
                         output_size=output_size,
                         activation=activation,
                         last_activation=last_activation)
        
## Theta Model

class ThetaModel:

    def __init__(self, 
                 input_norm_params=None,
                 output_norm_params=None,
                 **kwargs):
        
        self.build_theta_model()
        self.input_norm_fn, self.input_unnorm_fn = self._compile_input_normalization_fn(norm_params=input_norm_params)
        self.output_norm_fn, self.output_unnorm_fn = self._compile_output_normalization_fn(norm_params=output_norm_params)

    def build_theta_model(self, **kwargs):
        self.theta_model = {}  
        
    def _compile_input_normalization_fn(self, norm_params=None, **kwargs):
        if norm_params is None:
            return lambda x: x, lambda x: x
        else:
            # to [-1, 1]
            min_val = norm_params["min"]
            max_val = norm_params["max"]
            norm_fn = lambda x: (x - min_val) / (max_val - min_val) * 2 - 1
            unnorm_fn = lambda x: (x + 1) / 2 * (max_val - min_val) + min_val
            return norm_fn, unnorm_fn
        
    def _compile_output_normalization_fn(self, norm_params=None, **kwargs):
        if norm_params is None:
            return lambda x: x, lambda x: x
        else:
            def output_norm_fn(theta_values):
                new_theta_values = {}
                for key, value in theta_values.items():
                    min_value = norm_params[key][0]
                    max_value = norm_params[key][1]
                    new_theta_values[key] = (value - min_value) / (max_value - min_value) * 2 - 1
                return new_theta_values
            def output_unnorm_fn(theta_values):
                new_theta_values = {}
                for key, value in theta_values.items():
                    min_value = norm_params[key][0]
                    max_value = norm_params[key][1]
                    new_theta_values[key] = (value + 1) / 2 * (max_value - min_value) + min_value
                return new_theta_values
            return output_norm_fn, output_unnorm_fn
    
    def get_weights(self,):
        weight_values = {}
        for key, theta in self.theta_model.items():
            weight_values[key] = {}
            for key_weight, weight in theta.state_dict().items():
                weight_values[key][key_weight] = weight.clone()
        return weight_values
    
    def set_weights(self, weight_values):
        for key, value in weight_values.items():
            self.theta_model[key].load_state_dict(value)
    
    def input_norm_fn(self, t):
        # to [-1, 1]
        return (t - self.T_domain[0]) / (self.T_domain[1] - self.T_domain[0]) * 2 - 1
    
    def output_unnorm_fn(self, theta_values):
        return theta_values
    
    def output_norm_fn(self, theta_values):
        return theta_values

    def parameters(self,):
        paramters = []
        for key in self.theta_model:
            paramters += list(self.theta_model[key].parameters())
        return paramters
    
    def forward(self, t, unnormalize=False):
        t = self.input_norm_fn(t)
        theta_values = {}
        for key in self.theta_model:
            theta_values[key] = self.theta_model[key](t)
        if unnormalize:
            theta_values = self.output_unnorm_fn(theta_values)
        return theta_values
    
## Mosquito

class MosquitoThetaModel(ThetaModel):
        
    def build_theta_model(self, **kwargs):
        
        self.alpha = torch.tensor(38.0)
        self.temperature = ConstantVariable(torch.tensor(9.0))
    
        self.theta_model = {}
        self.theta_model['gamma_Aem'] = ConstantVariable(torch.tensor(1.143))
        self.theta_model['gamma_Ab'] = ConstantVariable(torch.tensor(0.885))
        self.theta_model['gamma_Ao'] = ConstantVariable(torch.tensor(2.0))
        self.theta_model['f_E'] = LambdaLayer(lambda temp: torch.maximum(torch.tensor(0.0), 0.16 * (torch.exp(0.105 * (temp-10)) - torch.exp(0.105*(self.alpha-10) - (self.alpha-temp)/5.007))))
        self.theta_model['f_P'] = LambdaLayer(lambda temp: torch.maximum(torch.tensor(0.0), 0.021 * (torch.exp(0.162 * (temp-10)) - torch.exp(0.162*(self.alpha-10) - (self.alpha-temp)/5.007))))
        self.theta_model['f_Ag'] = LambdaLayer(lambda temp: torch.maximum(torch.tensor(0.0), temp-9.8)/64.4)
        self.theta_model['mu_E'] = ConstantVariable(torch.tensor(0.0))
        self.theta_model['mu_L'] = ConstantVariable(torch.tensor(0.0304))
        self.theta_model['mu_P'] = ConstantVariable(torch.tensor(0.0146))
        self.theta_model['mu_em'] = ConstantVariable(torch.tensor(0.1))
        self.theta_model['mu_r'] = ConstantVariable(torch.tensor(0.08))
        self.theta_model['mu_A'] = ConstantVariable(torch.tensor(1/43)) 
        self.theta_model['kappa_L'] = ConstantVariable(torch.tensor(8e8))
        self.theta_model['kappa_P'] = ConstantVariable(torch.tensor(1e7))
        self.theta_model['sigma'] = ConstantVariable(torch.tensor(0.5))
        self.theta_model['beta_1'] = ConstantVariable(torch.tensor(141))
        self.theta_model['beta_2'] = ConstantVariable(torch.tensor(80))

        self.theta_model['m_A'] = LambdaLayer(lambda temp: torch.maximum(torch.tensor(-0.005941 + 0.002965 * temp), self.theta_model['mu_A'](temp)))
        self.theta_model['f_L'] = LambdaLayer(lambda temp: self.theta_model['f_P'](temp) / 1.65)
        self.theta_model['m_E'] = LambdaLayer(lambda temp: self.theta_model['mu_E'](temp))
        self.theta_model['m_L'] = LambdaLayer(lambda temp: self.theta_model['mu_L'](temp) + torch.exp(-temp/2))
        self.theta_model['m_P'] = LambdaLayer(lambda temp: self.theta_model['mu_P'](temp) + torch.exp(-temp/2))

class SinMosquitoThetaModel(MosquitoThetaModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)        
        self.temperature = LambdaLayer(lambda t: 10 * torch.sin(2 * np.pi / 365 * t) + 10) # 10 * np.sin(2 * np.pi / 365 * t) + 10

    def forward(self, t, unnormalize=True):
        temperature = self.temperature(t)
        theta_values = {}
        for key in self.theta_model:
            if key == "temperature":
                continue
            theta_values[key] = self.theta_model[key](temperature)
        return theta_values
    
class MosquitoPINN(CausalTrainingGradientBalancingODePINN):

    def build_u_model(self, **kwargs):
        self.u_model = MLP_UModel(input_size=1, 
                                output_size=10,
                                hidden_layer_sizes=[100, 100, 100, 100],
                                activation=nn.GELU(),
                                last_activation=nn.Identity(),
                                input_norm_params=self.input_norm_params,
                                output_norm_params=self.output_norm_params,)
        
    def build_theta_model(self, **kwargs):
        self.theta_model = SinMosquitoThetaModel()

    def ode_system(self, u_values, theta_values, **kwargs):

        E = u_values[:, 0].unsqueeze(-1)
        L = u_values[:, 1].unsqueeze(-1)
        P = u_values[:, 2].unsqueeze(-1)
        Aem = u_values[:, 3].unsqueeze(-1)
        Ab1 = u_values[:, 4].unsqueeze(-1)
        Ag1 = u_values[:, 5].unsqueeze(-1)
        Ao1 = u_values[:, 6].unsqueeze(-1)
        Ab2 = u_values[:, 7].unsqueeze(-1)
        Ag2 = u_values[:, 8].unsqueeze(-1)
        Ao2 = u_values[:, 9].unsqueeze(-1)

        dE_dt = theta_values['gamma_Ao'] * (theta_values['beta_1'] * Ao1 + theta_values['beta_2'] * Ao2) - (theta_values['mu_E'] + theta_values['f_E']) * E
        dL_dt = theta_values['f_E'] * E - (theta_values['m_L'] * (1 + L / theta_values['kappa_L']) + theta_values['f_L']) * L
        dP_dt = theta_values['f_L'] * L - (theta_values['m_P'] + theta_values['f_P']) * P
        dAem_dt = theta_values['f_P'] * P * theta_values['sigma'] * torch.exp(- theta_values['mu_em'] * (1 + P / theta_values['kappa_P'])) \
            - (theta_values['m_A'] + theta_values['gamma_Aem']) * Aem
        dAb1_dt = theta_values['gamma_Aem'] * Aem - (theta_values['m_A'] + theta_values['mu_r'] + theta_values['gamma_Ab']) * Ab1
        dAg1_dt = theta_values['gamma_Ab'] * Ab1 - (theta_values['m_A'] + theta_values['f_Ag']) * Ag1
        dAo1_dt = theta_values['f_Ag'] * Ag1 - (theta_values['m_A'] + theta_values['mu_r'] + theta_values['gamma_Ao']) * Ao1
        dAb2_dt = theta_values['gamma_Ao'] * (Ao1 + Ao2) - (theta_values['m_A'] + theta_values['mu_r'] + theta_values['gamma_Ab']) * Ab2
        dAg2_dt = theta_values['gamma_Ab'] * Ab2 - (theta_values['m_A'] + theta_values['f_Ag']) * Ag2
        dAo2_dt = theta_values['f_Ag'] * Ag2 - (theta_values['m_A'] + theta_values['mu_r'] + theta_values['gamma_Ao']) * Ao2

        return [dE_dt, dL_dt, dP_dt, dAem_dt, dAb1_dt, dAg1_dt, dAo1_dt, dAb2_dt, dAg2_dt, dAo2_dt]






















## Lorenz


class LorenzThetaModel(ThetaModel):

    def build_theta_model(self, **kwargs):
        self.theta_model = {}
        self.theta_model['sigma'] = ConstantVariable(torch.tensor(10))
        self.theta_model['rho'] = ConstantVariable(torch.tensor(28))
        self.theta_model['beta'] = ConstantVariable(torch.tensor(8/3))

class LorenzOdePINN(CausalTrainingGradientBalancingODePINN):

    def build_u_model(self, **kwargs):
        self.u_model = MLP_UModel(input_size=1, 
                                output_size=3,
                                hidden_layer_sizes=[100, 100, 100, 100],
                                activation=nn.GELU(),
                                last_activation=nn.Identity(),
                                input_norm_params=self.input_norm_params,
                                output_norm_params=self.output_norm_params,)

    def build_theta_model(self, **kwargs):
        self.theta_model = LorenzThetaModel()

    def ode_system(self, u_values, theta_values):
        unnormed_x = u_values[:, 0].unsqueeze(-1)
        unnormed_y = u_values[:, 1].unsqueeze(-1)
        unnormed_z = u_values[:, 2].unsqueeze(-1)

        dx_dt = theta_values['sigma'] * (unnormed_y - unnormed_x)
        dy_dt = unnormed_x * (theta_values['rho'] - unnormed_z) - unnormed_y
        dz_dt = unnormed_x * unnormed_y - theta_values['beta'] * unnormed_z
        return dx_dt, dy_dt, dz_dt

class InverseLorenzThetaModel(ThetaModel):

    def build_theta_model(self, **kwargs):
        self.theta_model = {}
        self.theta_model['sigma'] = MLP(hidden_layer_sizes=(32,32,32), activation=nn.GELU(), last_activation=nn.Identity())
        self.theta_model['rho'] = MLP(hidden_layer_sizes=(32,32,32), activation=nn.GELU(), last_activation=nn.Identity())
        self.theta_model['beta'] = MLP(hidden_layer_sizes=(32,32,32), activation=nn.GELU(), last_activation=nn.Identity())

class InverseLorenzOdePINN(LorenzOdePINN):

    def build_theta_model(self, **kwargs):
        self.theta_model = InverseLorenzThetaModel(input_norm_params=self.input_norm_params,
                                      output_norm_params=self.theta_norm_params,
                                      **kwargs)
