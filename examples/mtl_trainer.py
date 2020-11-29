"""
    The PyTorch implementation of Schrodinger-ANI

    Tensorboard logs, a copy of trainer.py, and model checkpoints are written
    in ./logs/YYYYMMDD_TIME/
"""
import sys
import torch
import datetime
import shutil
import math
import tqdm
import random
from torch.utils import tensorboard

sys.path.append('../')
try:
    from nn import RNN_S, EMA, MTLLoss, ActivationHistogram
    from data import anidata_loader
    from aev import AEVComputer
    from optim import AdamaxW
except ImportError:
    raise

torch.manual_seed(0)
random.seed(0)

# device to run the training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'GPU allocated: {torch.cuda.get_device_name(0)}')

###############################################################################
HATREE2KCALMOL = 627.509469
Rcr = 5.2000e+00
Rca = 3.5000e+00
EtaR = torch.tensor([1.6000000e+01], device=device)
ShfR = torch.tensor([9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00, 3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00], device=device)
Zeta = torch.tensor([3.2000000e+01], device=device)
ShfZ = torch.tensor([1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00], device=device)
EtaA = torch.tensor([8.0000000e+00], device=device)
ShfA = torch.tensor([9.0000000e-01, 1.5500000e+00, 2.2000000e+00, 2.8500000e+00], device=device)

network_dims = [
    [160, 128, 96],  # H
    [144, 112, 96],  # C
    [128, 112, 96],  # N
    [128, 112, 96],  # O
    [128, 112, 96],  # S
    [128, 112, 96],  # F
    [128, 112, 96],  # Cl
    [128, 112, 96],  # P
]
num_species = len(network_dims)
aev_computer = AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species).to(device)

###############################################################################
batch_size = 256
data_path = '../../charged_data/'

# for train_* upto round 5
data_stat = {
    -3: (0.42898974, 0.03999415),
    -2: (0.17614624, 0.11558004),
    -1: (0.10169958, 0.13883956),
    0: (0.052970566, 0.05525185),
    1: (0.35364136, 0.0914877),
    2: (0.76599836, 0.095402874),
    3: (1.376284, 0.18854873),
    4: (1.8260088, 0.026283875)
}
training = anidata_loader(data_path + 'train_*', ('charges',)).remove_outliers(data_stat).shuffle()
validation = anidata_loader(data_path + 'valid_?', ('charges',)).remove_outliers(data_stat).shuffle()
validationq = anidata_loader(data_path + 'valid_ionic*', ('charges',)).remove_outliers(data_stat).shuffle()
test = anidata_loader(data_path + 'test_*', ('charges',)).shuffle()

training = training.cache()
validation = validation.collate(batch_size).cache()
validationq = validationq.collate(batch_size).cache()
test = test.collate(batch_size).cache()

###############################################################################
model = RNN_S(aev_computer, network_dims, chinet_dims=[160, 128, 96]).to(device)
print(model)


###############################################################################
# Initialize the weights and biases.
def kaiming_init(m):
    """He kaiming Normal initialization"""

    if isinstance(m, torch.nn.Linear):
        fan_out, fan_in = m.weight.shape
        data_std = 1.0  # makes NN outputs have similar stddev to dat
        slope = 0.0
        if fan_out == 3:  # last layer
            data_std = 0.1
            slope = 1.0

        std = torch.sqrt(
            (
                torch.as_tensor(2.0) / ((1 + slope**2) * float(fan_in))
            )
        )

        torch.nn.init.normal_(m.weight, mean=0.0, std=std * data_std)
        torch.nn.init.zeros_(m.bias)


model.ani_nets.apply(kaiming_init)

###############################################################################
# separate weights and biases to assign weight decay in optim
model_weights = []
model_biases = []
for name, param in model.named_parameters():
    if 'bias' in name:
        model_biases.append(param)
    elif 'weight' in name:
        model_weights.append(param)

assert len(list(model.parameters())) == len(model_biases) + len(model_weights)

optimizer = AdamaxW([
    {'params': model_weights, 'weight_decay': 1e-4},
    {'params': model_biases},
], lr=1e-3)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.98, patience=1, threshold=0)

###############################################################################
# Setting up directories and checkpoint files to store checkpoints and logs
now = datetime.datetime.now()
date = now.strftime("%Y%m%d_%H%M")
log = 'logs/' + str(date)

training_writer = tensorboard.SummaryWriter(log_dir=log + '/train')

# save a copy of executed script
shutil.copy(__file__, log + '/trainer.py')

latest_checkpoint = log + '/latest.pt'
best_model_checkpoint = log + '/best_'


def save_model(checkpoint):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }, checkpoint)


###############################################################################
# Validation function
def validate(dataset):
    with torch.no_grad():
        # run validation
        mse_sum = torch.nn.MSELoss(reduction='sum')
        energy_mse = 0.0
        charge_mse = 0.0
        count = 0
        atom_count = 0
        for properties in dataset:
            species = properties['species'].to(device)
            charges = properties['charges'].to(device)
            net_charges = charges.sum(dim=1)

            coordinates = properties['coordinates'].to(device).float()
            true_energies = properties['energies'].to(device).float()
            _, predicted_energies, predicted_charges = model(
                (species, coordinates, net_charges)
            )
            energy_mse += mse_sum(predicted_energies, true_energies).item()
            charge_mse += mse_sum(predicted_charges.flatten(), charges.flatten()).item()
            count += predicted_energies.shape[0]
            atom_count += (species >= 0).sum(dtype=true_energies.dtype)
    return math.sqrt(energy_mse / count) * HATREE2KCALMOL, math.sqrt(charge_mse / atom_count)


###############################################################################
# Training loop.
mtl = MTLLoss(num_tasks=2).to(device)
optimizer.param_groups[0]['params'].append(mtl.log_sigma)

mse = torch.nn.MSELoss(reduction='none')
ema = EMA(model, decay=0.999)

max_epochs = 500
early_stopping_learning_rate = 1.0E-5

for epoch in range(max_epochs):
    # shuffle training data
    training = training.shuffle()
    ds = training.collate(batch_size)

    ema.assign(model)

    # rmse using EMA of model params
    ema_rmse, charge_rmse = validate(validation)
    ema_rmseq, charge_rmseq = validate(validationq)
    ema_rmset, charge_rmset = validate(test)

    # save best checkpoint (scheduler updated with rmse)
    # if scheduler.is_better(ema_rmse, scheduler.best):
    if epoch > 5:
        save_model(best_model_checkpoint + f'{epoch}.pt')

    ema.resume(model)

    training_writer.add_scalar('vrmse', ema_rmse, epoch)
    training_writer.add_scalar('charge_vrmse', charge_rmse, epoch)
    training_writer.add_scalar('vrmseq', ema_rmseq, epoch)
    training_writer.add_scalar('charge_vrmseq', charge_rmseq, epoch)
    training_writer.add_scalar('vrmset', ema_rmset, epoch)
    training_writer.add_scalar('charge_vrmset', charge_rmset, epoch)
    print(f'EMA_RMSE: {ema_rmse:.4} at epoch {epoch}')

    learning_rate = optimizer.param_groups[0]['lr']
    training_writer.add_scalar('learning_rate', learning_rate, epoch)
    if learning_rate < early_stopping_learning_rate:
        break

    scheduler.step(ema_rmse)

    for i, properties in tqdm.tqdm(
        enumerate(ds),
        total=len(ds),
        desc="epoch {}".format(epoch),
        disable=False
    ):
        # with torch.autograd.set_detect_anomaly(True):
        species = properties['species'].to(device)
        charges = properties['charges'].to(device).float()
        net_charges = charges.sum(dim=1)

        coordinates = properties['coordinates'].to(device).float()
        true_energies = properties['energies'].to(device).float()
        num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
        _, predicted_energies, predicted_charges = model(
            (species, coordinates, net_charges)
        )

        # Errors are weighted by exp( -(absolute energy per atom) / kT)
        # clipped to range. Absolute energy should be approximately
        # zero-centered for equilibrium geometries
        exp_weight = torch.exp(- true_energies / num_atoms / 0.006).clamp_(0.01, 1.0)
        energy_loss = (mse(predicted_energies, true_energies) * exp_weight).sum()
        charge_loss = (mse(predicted_charges, charges) * exp_weight.unsqueeze(-1)).sum() / num_atoms.sum()

        loss = mtl(energy_loss, charge_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema(model)  # update EMA of model params

        # additional tensorboard writers to monitor training
        iteration = epoch * len(ds) + i
        training_writer.add_scalar('batch_loss', loss, iteration)
        training_writer.add_scalar('precision_e', mtl.get_precisions()[0], iteration)
        training_writer.add_scalar('precision_q', mtl.get_precisions()[1], iteration)

        if iteration % 40000 == 0:

            # writing the weights and biases
            for label, m in model.named_modules():
                if isinstance(m, ActivationHistogram):
                    activation = torch.cat(m.activation_handles, dim=0)
                    training_writer.add_histogram(
                        str(label) + "_" + str(m.name),
                        activation,
                        iteration
                    )
            for name, param in model.named_parameters():
                training_writer.add_histogram(name, param.data.cpu().numpy(), iteration)

                # writing gradients, and their std
                if param.grad is not None:
                    training_writer.add_histogram(
                        str(name) + "_grad", param.grad.data.cpu().numpy(), iteration
                    )
                    training_writer.add_scalar(
                        str(name) + "_grad_stddev", torch.std(
                            param.grad.data, unbiased=False
                        ).cpu().numpy(), iteration
                    )
    save_model(latest_checkpoint)
