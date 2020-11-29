import os
import sys
import torch
import math
import copy
import numpy as np

import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining

sys.path.append('../')
try:
    from nn import SANIModel, EMA
    from data import anidata_loader
    from aev import AEVComputer
    from optim import AdamaxW
except ImportError:
    raise

# torch.manual_seed(0)
# random.seed(0)

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
aev_dim = aev_computer.aev_length

training = anidata_loader('/home/farhad/scratch/Schrodinger/data/*train*').shuffle()
validation = anidata_loader('/home/farhad/scratch/Schrodinger/data/*valid*').shuffle()

training = training.cache()
validation = validation.collate(2048).cache()


###############################################################################
def atomic_net(layers_dim):
    assert len(layers_dim) == 3
    return torch.nn.Sequential(
        torch.nn.Linear(aev_dim, layers_dim[0]),
        torch.nn.CELU(0.1),
        torch.nn.Linear(layers_dim[0], layers_dim[1]),
        torch.nn.CELU(0.1),
        torch.nn.Linear(layers_dim[1], layers_dim[2]),
        torch.nn.CELU(0.1),
        torch.nn.Linear(layers_dim[2], 1)
    )


###############################################################################
# Initialize the weights and biases.
def kaiming_init(m):
    """He kaiming Normal initialization"""

    if isinstance(m, torch.nn.Linear):
        fan_out, fan_in = m.weight.shape
        data_std = 1.0  # makes NN outputs have similar stddev to dat
        slope = 0.0
        if fan_out == 1:  # last layer
            data_std = 0.1
            slope = 1.0

        std = torch.sqrt(
            (
                torch.as_tensor(2.0) / ((1 + slope**2) * float(fan_in))
            )
        )

        torch.nn.init.normal_(m.weight, mean=0.0, std=std * data_std)
        torch.nn.init.zeros_(m.bias)


###############################################################################
# Validation function
def validate(model, dataset):
    with torch.no_grad():
        # run validation
        mse_sum = torch.nn.MSELoss(reduction='sum')
        total_mse = 0.0
        count = 0
        for properties in dataset:
            species = properties['species'].to(device)
            coordinates = properties['coordinates'].to(device).float()
            true_energies = properties['energies'].to(device).float()
            _, aevs = aev_computer((species, coordinates))
            _, predicted_energies = model((species, aevs))
            total_mse += mse_sum(predicted_energies, true_energies).item()
            count += predicted_energies.shape[0]
    return math.sqrt(total_mse / count) * HATREE2KCALMOL


def get_wb(model):
    """separate weights and biases to assign weight decay in optim"""
    model_weights = []
    model_biases = []
    for name, param in model.named_parameters():
        if 'bias' in name:
            model_biases.append(param)
        elif 'weight' in name:
            model_weights.append(param)
    return model_weights, model_biases


class PytorchTrainble(tune.Trainable):
    """Train a Pytorch SANI with Trainable and PopulationBasedTraining
       scheduler.
    """
    def _setup(self, config):
        self.batch_size = config.get("batch_size", 256)
        self.training = training
        self.model = SANIModel(
            [atomic_net(layers_dim) for layers_dim in network_dims]
        ).to(device)
        self.model.apply(kaiming_init)

        model_weights, model_biases = get_wb(self.model)
        self.optimizer = AdamaxW([
            {
                'params': model_weights,
                'weight_decay': config.get('weight_decay', 1e-4)},
            {'params': model_biases},
        ], lr=config.get("lr", 1e-3))

        self.ema = EMA(self.model, decay=0.999)

    def _train(self):
        mse = torch.nn.MSELoss(reduction='none')
        self.training = self.training.shuffle()
        ds = self.training.collate(self.batch_size)

        for i, properties in enumerate(ds):

            species = properties['species'].to(device)
            coordinates = properties['coordinates'].to(device).float()
            true_energies = properties['energies'].to(device).float()
            num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
            _, aevs = aev_computer((species, coordinates))
            _, predicted_energies = self.model((species, aevs))

            exp_weight = torch.exp(
                - true_energies / num_atoms / 0.006
            ).clamp_(0.01, 1.0)
            loss = (mse(predicted_energies, true_energies) * exp_weight).sum()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.ema(self.model)

        self.ema.assign(self.model)
        ema_rmse = validate(self.model, validation)
        self.ema.resume(self.model)

        return {"ema_rmse": ema_rmse}

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")

        # get the param ema state_dict()
        self.ema.assign(self.model)
        shadow = copy.deepcopy(self.model.state_dict())
        self.ema.resume(self.model)

        torch.save({
            "model": self.model.state_dict(),
            "shadow": shadow,
            "optimizer": self.optimizer.state_dict(),
        }, checkpoint_path)

        return checkpoint_path

    def _restore(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)

        # re-initialize EMA with params shadows
        self.model.load_state_dict(checkpoint["shadow"])
        self.ema = EMA(self.model, decay=0.999)

        # load model params
        self.model.load_state_dict(checkpoint["model"])

        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def reset_config(self, new_config):
        del self.optimizer

        model_weights, model_biases = get_wb(self.model)
        self.optimizer = AdamaxW([
            {
                'params': model_weights,
                'weight_decay': new_config.get('weight_decay', 1e-4)},
            {'params': model_biases},
        ], lr=new_config.get("lr", 1e-3))

        self.config = new_config
        return True


ray.shutdown()  # Restart Ray defensively in case the ray connection is lost.
ray.init(log_to_driver=False)

scheduler = PopulationBasedTraining(
    time_attr="training_iteration",
    metric="ema_rmse",
    mode="min",
    perturbation_interval=5,
    hyperparam_mutations={
        # distribution for resampling
        "lr": lambda: np.random.uniform(1e-4, 1e-3),
        "batch_size": [64, 128, 256, 512, 1024],
        "weight_decay": [1e-3, 1e-4, 1e-5, 1e-6],
    }
)

ray.shutdown()  # Restart Ray defensively in case the ray connection is lost.
ray.init(log_to_driver=False)

analysis = tune.run(
    PytorchTrainble,
    name="pbt_test",
    scheduler=scheduler,
    checkpoint_freq=5,
    checkpoint_at_end=True,
    reuse_actors=True,
    verbose=1,
    stop={
        "training_iteration": 500,
    },
    num_samples=5,

    # PBT starts by training many neural networks in
    # parallel with random hyperparameters.
    config={
        "lr": tune.uniform(1e-4, 1e-3),
    },
    resources_per_trial={"cpu": 4, 'gpu': 1},
    local_dir="./pbt_results",
)
print("Best config: ", analysis.get_best_config(metric="ema_rmse"))
