# %%

import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from neuralop.models import FNO
from neuralop.training import AdamW
from neuralop import H1Loss, LpLoss
from neuralop.utils import count_model_params

from datetime import datetime

import optuna
optuna_db_name = 'db.sqlite3'
class emSHG_Dataset(Dataset):
    def __init__(self, input_series, output_series):
        self.input_series = input_series
        self.output_series = output_series

    def __len__(self):
        return len(self.input_series)

    def __getitem__(self, idx):
        return self.input_series[idx], self.output_series[idx]


def create_dataloaders(file_path, samples=1000, batch_size=128, test_size=0.1, use_truncation=False, values=None, use_fft=False, use_embeddings=None, embed_freqs=7):

    data = np.load(file_path)
    
    poling_region_length = torch.from_numpy(data['poling_region_length'])[:samples].to(torch.complex64)
    poling_period_mismatch = torch.from_numpy(data['poling_period_mismatch'])[:samples].to(torch.complex64)
    pump_energy = torch.from_numpy(data['pump_energy'])[:samples].to(torch.complex64)
    input_field = torch.from_numpy(data['input_field'])[:samples].to(torch.complex64)
    output_field = torch.from_numpy(data['output_field'])[:samples].to(torch.complex64)
    
    if use_truncation:
        num = len(values)
    else:
        num = output_field.shape[-1]
        values = [i for i in range(0, num)]
    

    if use_embeddings:  
        input_data = torch.zeros(samples, (2*embed_freqs+1)*3+1, num, dtype=torch.complex64)
    else:
        input_data = torch.zeros((samples, 4, num), dtype=torch.complex64)
    
    input_data[:, 0, :] = poling_region_length.repeat(num,1).permute(1,0)
    input_data[:, 1, :] = poling_period_mismatch.repeat(num,1).permute(1,0)
    input_data[:, 2, :] = pump_energy.repeat(num,1).permute(1,0)
    
    if use_embeddings:
        t = torch.linspace(0, 2*torch.pi, num, dtype=torch.complex64)
        for i in range(0,embed_freqs):
            input_data[:, 3+6*i, :] = poling_region_length.repeat(num,1).permute(1,0)  * np.cos((i+1)*t).repeat(samples,1)
            input_data[:, 4+6*i, :] = poling_region_length.repeat(num,1).permute(1,0)  * np.sin((i+1)*t).repeat(samples,1)
            input_data[:, 5+6*i, :] = poling_period_mismatch.repeat(num,1).permute(1,0) * np.cos((i+1)*t).repeat(samples,1)
            input_data[:, 6+6*i, :] = poling_period_mismatch.repeat(num,1).permute(1,0) * np.sin((i+1)*t).repeat(samples,1)
            input_data[:, 7+6*i, :] = pump_energy.repeat(num,1).permute(1,0) * np.cos((i+1)*t).repeat(samples,1)
            input_data[:, 8+6*i, :] = pump_energy.repeat(num,1).permute(1,0) * np.sin((i+1)*t).repeat(samples,1)
    
    input_data[:, -1, :] = input_field[:,values]
    
    output_data = output_field[:, values].unsqueeze(1)
    
    if use_fft:
        input_data[:, -1, :] = torch.fft.fft(input_data[:, -1, :], dim=-1)
        output_data = torch.fft.fft(output_data, dim=-1)
    
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=test_size, random_state=42)

    # Create Dataset objects
    train_dataset = emSHG_Dataset(X_train, y_train)
    test_dataset = emSHG_Dataset(X_test, y_test)

    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader



if __name__ == '__main__':
    print()
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    device = 'cuda'
 
    # %% ###############################################################

    file_path = "/global/cfs/cdirs/m4505/tensorlab/EM_Dataset/emSHG_data_SingleCPU.npz"    
    num_samples = 10000
    test_size = 0.2
    
    use_truncation = True
    truncation_values = [i for i in range(600, 1300)]
    
    use_fft = False
    
    L2loss = LpLoss(d=1, p=2, reduce_dims=[0,1], reductions=['sum', 'mean'])
    H1loss = H1Loss(d=1, reduce_dims=[0,1], reductions=['sum', 'mean'])
    
    
    
    def objective(trial):
        
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        epochs = 2000  
        
        n_modes = 512
        hidden_channels = 256
        projection_channels = 256
        
        abs_or_rel = trial.suggest_categorical('abs_or_rel', ['abs', 'rel'])
        
        loss_type = trial.suggest_categorical('loss_type', ['L2', 'H1'])
        if loss_type == 'L2':
            train_loss = L2loss
        elif loss_type == 'H1':
            train_loss = H1loss
        
        n_layers = trial.suggest_int('n_layers', 8, 16)
        
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])
        
        weight_decay_power = trial.suggest_int('weight_decay_power', 2, 8)
        weight_decay = 10**(-weight_decay_power)
        
        lr = trial.suggest_categorical('lr_val', [8e-5, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4])
        
        
        use_embeddings = trial.suggest_categorical('use_embeddings', [True, False])
        if use_embeddings:
            embed_freqs = trial.suggest_int('embed_d', 1, 32)
        else:
            embed_freqs = -1

        train_loader, test_loader = create_dataloaders(file_path, 
                                                    samples=num_samples, batch_size=batch_size, test_size=test_size, 
                                                    use_truncation=use_truncation, values=truncation_values,
                                                    use_fft=use_fft, 
                                                    use_embeddings=use_embeddings, embed_freqs=embed_freqs)

        # Example data
        input_data, _ = next(iter(train_loader))
        
        model = FNO(n_modes=(n_modes,), in_channels=input_data.shape[1], out_channels=1, hidden_channels=hidden_channels, projection_channels=projection_channels, n_layers=n_layers, complex_spatial_data=True, domain_padding=None) 
        model = model.to(device)
       
        number_params = count_model_params(model)
        trial.set_user_attr("number_params", number_params)
        min_number_params = 1e6
        max_number_params = 1e12
        if number_params < min_number_params:
            return 200
        if number_params > max_number_params:
            return 200
        
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        patience = trial.suggest_int('patience', 8, 24)
        scheduler_factor =  0.8
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = scheduler_factor, patience=patience)
        
        for ep in range(epochs):
            model.train()
            train_H1_abs_loss = 0
            train_L2_abs_loss = 0
            train_H1_rel_loss = 0
            train_L2_rel_loss = 0
            
            train_loss_val = 0
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                out = model(x)
                
                if abs_or_rel == 'abs':
                    loss = train_loss.abs(out, y)
                elif abs_or_rel == 'rel':
                    loss = train_loss.rel(out, y)
                
                train_loss_val += loss.item()
                
                train_H1_abs_loss += H1loss.abs(out, y).item()
                train_L2_abs_loss += L2loss.abs(out, y).item()
                train_H1_rel_loss += H1loss(out, y).item()
                train_L2_rel_loss += L2loss(out, y).item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            scheduler.step(train_loss_val)
            
            test_H1_abs_loss = 0
            test_L2_abs_loss = 0
            test_H1_rel_loss = 0
            test_L2_rel_loss = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(device)
                    y = y.to(device)
                    out = model(x)
                    
                    test_H1_abs_loss += H1loss.abs(out, y).item()
                    test_L2_abs_loss += L2loss.abs(out, y).item()
                    test_H1_rel_loss += H1loss(out, y).item()
                    test_L2_rel_loss += L2loss(out, y).item()
            
            train_H1_abs_loss = train_H1_abs_loss/len(train_loader.dataset)
            train_L2_abs_loss = train_L2_abs_loss/len(train_loader.dataset)
            train_H1_rel_loss = train_H1_rel_loss/len(train_loader.dataset)
            train_L2_rel_loss = train_L2_rel_loss/len(train_loader.dataset)
            
            test_H1_abs_loss = test_H1_abs_loss/len(test_loader.dataset)
            test_L2_abs_loss = test_L2_abs_loss/len(test_loader.dataset)
            test_H1_rel_loss = test_H1_rel_loss/len(test_loader.dataset)
            test_L2_rel_loss = test_L2_rel_loss/len(test_loader.dataset)
            
            trial.report(train_L2_rel_loss, ep)
            
            trial.set_user_attr(f"epoch_{ep}_train_L2_H1_test_L2_H1_abs", (f"{train_L2_abs_loss:.3f}", f"{train_H1_abs_loss:.3f}", f"{test_L2_abs_loss:.3f}", f"{test_H1_abs_loss:.3f}"))
            trial.set_user_attr(f"epoch_{ep}_train_L2_H1_test_L2_H1_rel", (f"{train_L2_rel_loss:.3f}", f"{train_H1_rel_loss:.3f}", f"{test_L2_rel_loss:.3f}", f"{test_H1_rel_loss:.3f}"))
            
            if ep>15 and train_L2_rel_loss>2:
                return 10
            if ep>30 and train_L2_rel_loss>1.01:
                return 2
            if ep>50 and train_L2_rel_loss>0.999:
                return 1
            
        return train_L2_rel_loss
    
    study = optuna.create_study(
        storage = "sqlite:///"+optuna_db_name+"?timeout=30000",  
        study_name = "Robert_Truncated_New",
        load_if_exists = True,
        direction = "minimize",
        sampler = optuna.samplers.RandomSampler(),
    )
    study.optimize(objective, n_trials=50)
    print(f"Best value: {study.best_value} (params: {study.best_params})")