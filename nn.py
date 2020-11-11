import math
import torch
import numpy as np
from aev import compute_radial_qaev

PI = np.pi
a0 = 0.529177249  # Bohr radii

# SIGMA extracted from ./tools/grimme_sigma.csv
SIGMA = torch.tensor([
    0.5515909, 1.8886297, 1.3225029, 1.2316629,
    2.1884933, 1.7750372, 1.3677907, 1.3820058
], dtype=torch.float32)  # A.U.


def elect_screen(distances, a=2.2, b=8.5):
    # plot 1/(1+e^((-x+2.2)*8.5)), 0.5(1-cos(pi*(x-1.69)/0.85))
    return 1.0 / (1.0 + torch.exp((- distances + a) * b))


# Helper functions
def nonzero_batch_dmat(coordinates, eps=1e-16):
    """
    Calculate distance matrix in atomic unit and replace zero elements by eps
    for a given batch coordinates.
    https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3

    Arguments:
        coordinates (torch.Tensor): (N, n, 3)
        eps (float): to avoid devision by zeros
    Returns:
        distances (torch.Tensor): (N, n, n)
    """
    assert len(coordinates.size()) == 3
    diff = coordinates.unsqueeze(1) - coordinates.unsqueeze(2)
    distances = torch.sqrt(torch.sum(diff * diff, -1) + eps)
    # distances[distances < eps] = eps  # avoid infs -> 0 * infs = nan
    return distances / a0


def ESPerf(pair_padding_mask, distances, charges, species):
    """
    Calculated electric potential

    Arguments:
        padding_mask (torch.Tensor): (N, n)
        distances (torch.Tensor): (N, n, n)
        charges (torch.Tensor): (N, n)
    Returns:
        electric potential (torch.Tensor): (N, n)
    """
    sig = SIGMA[species].to(species.device)

    # N, n, n tensor of sum of sigma squared
    sig_sqsum = sig.unsqueeze(1) ** 2 + sig.unsqueeze(2) ** 2
    sig_sqsum[sig_sqsum < 1e-8] = 1e-8

    j_ij = torch.erf(
        distances / torch.sqrt(2.0 * sig_sqsum)
    ) / distances

    esp_mat = charges.unsqueeze(-1) * j_ij
    diag_mask = (1 - torch.eye(*distances.shape[1:])).unsqueeze(0).to(distances.device)
    return (esp_mat * pair_padding_mask * diag_mask).sum(dim=-2)


def ESP(pair_padding_mask, distances, charges):
    """
    Calculated electric potential

    Arguments:
        padding_mask (torch.Tensor): (N, n)
        distances (torch.Tensor): (N, n, n)
        charges (torch.Tensor): (N, n)
    Returns:
        electric potential (torch.Tensor): (N, n)
    """
    esp_mat = charges.unsqueeze(-1) / distances
    diag_mask = (1 - torch.eye(*distances.shape[1:])).unsqueeze(0).to(distances.device)
    return (esp_mat * pair_padding_mask * diag_mask).sum(dim=-2)


def charge_transfer(pair_padding_mask, chi, coulomb, species, distances):
    """
    Calculated the pairwise charge transfer.

    Arguments:
        coulomb matrix (torch.Tensor): (N, n, n)
        chi (torch.Tensor): (N, n)
        distances (torch.Tensor): (N, n, n)
        pair_padding_mask (torch.Tensor): (N, n, n) to ignore
            pairs associated with padded species
    Returns:
        charge_transfered (torch.Tensor): (N, n)
    """
    # apply decay functiom (shifted 1s overlap)
    sig = SIGMA[species].to(species.device)
    sig_sum = sig.unsqueeze(1) + sig.unsqueeze(2)  # N, n, n
    R = distances - sig_sum
    decay = torch.exp(- R) * (1.0 + R + 1.0/3.0 * R**2)

    j_ii = torch.diagonal(coulomb, dim1=-2, dim2=-1)  # N, n
    chi_diff = chi.unsqueeze(1) - chi.unsqueeze(2)
    j_ii_products = j_ii.unsqueeze(1) * j_ii.unsqueeze(2)
    pair_transfer = decay * (j_ii_products * chi_diff) * pair_padding_mask
    return torch.sum(pair_transfer, dim=-1)


def get_coulomb(pair_padding_mask, species, distances, eps=1e-8):
    """
    Electrostatic energy calculated from charge equilibrium method

    Arguments:
        pair_padding_mask: (N, n, n)
        chi (torch.Tensor): (N, n)
        species (torch.Tensor): (N, n)
        distances (torch.Tensor): (N, n, n)
        eps (float): to avoid devision by zeros
    Returns:
        coulomb matrix (torch.Tensor): (N, n, n)
    """
    sig = SIGMA[species].to(species.device)

    # N, n, n tensor of sum of sigma squared
    sig_sqsum = sig.unsqueeze(1) ** 2 + sig.unsqueeze(2) ** 2
    sig_sqsum[sig_sqsum < eps] = eps

    j_ij = torch.erf(
        distances / torch.sqrt(2.0 * sig_sqsum)
    ) / distances * pair_padding_mask

    j_ii = 1.0 / torch.sqrt(
        sig_sqsum * PI / 2.0
    ) * torch.eye(sig_sqsum.shape[-1]).to(sig_sqsum.device)

    coulomb = torch.triu(j_ij, diagonal=1) + j_ii
    return coulomb


class ActivationHistogram(torch.nn.Module):
    """
    Helper class to record the activation
    histogram data for tensorboard
    """
    def __init__(self, name):
        super(ActivationHistogram, self).__init__()
        self.activation_handles = []
        self.name = name

    def forward(self, inputs):
        self.activation_handles = []
        self.activation_handles.append(inputs)
        return inputs


class RNN_S(torch.nn.Module):

    def __init__(self, aev_computer, network_dims, chinet_dims):
        super(RNN_S, self).__init__()

        self.aev_computer = aev_computer

        def atomic_net(layers_dims, input_dim):
            assert len(layers_dims) == 3

            return torch.nn.Sequential(
                torch.nn.Linear(input_dim, layers_dims[0]),
                torch.nn.CELU(0.1),
                torch.nn.Linear(layers_dims[0], layers_dims[1]),
                torch.nn.CELU(0.1),
                torch.nn.Linear(layers_dims[1], layers_dims[2]),
                torch.nn.CELU(0.1),
                torch.nn.Linear(layers_dims[2], 1)
            )

        self.ani_nets = torch.nn.ModuleList(
            [
                atomic_net(
                    layers_dims, aev_computer.aev_length + aev_computer.radial_length + 2
                ) for layers_dims in network_dims
            ]
        )
        self.chi_net = torch.nn.ModuleList(
            [
                atomic_net(
                    layers_dims, aev_computer.aev_length + aev_computer.radial_length + 2
                ) for layers_dims in network_dims
            ]
        )

    def forward(self, model_input):
        species, coordinates, net_charge = model_input
        _, aev = self.aev_computer((species, coordinates))
        distances = nonzero_batch_dmat(coordinates)
        sig = SIGMA[species].to(species.device)
        jii = 1.0 / math.sqrt(PI) / sig

        mask = species.ne(-1)
        padding_mask = mask.float()
        pair_padding_mask = padding_mask.unsqueeze(1) * padding_mask.unsqueeze(2)

        species_ = species.flatten()
        aev = aev.flatten(0, 1)
        chi = aev.new_zeros(species_.shape)
        atomic_energies = aev.new_zeros(species_.shape)

        # init guess
        # approx with qnet / num_atoms, or linear fit?
        # pred_charges = init_charge_guess[species] * padding_mask
        # esp = ESPerf(pair_padding_mask, distances, pred_charges, species)
        pred_charges = aev.new_zeros(species.shape)
        esp = aev.new_zeros(species.shape)
        qraev = compute_radial_qaev(
            species,
            coordinates,
            pred_charges,
            self.aev_computer.constants(),
            self.aev_computer.sizes,
            None
        )

        # RNN
        for _ in range(2):
            input_features = torch.cat(
                (
                    aev,
                    qraev.flatten(0, 1),
                    pred_charges.flatten().unsqueeze(-1),
                    esp.flatten().unsqueeze(-1)
                ), dim=1
            )

            # chi_net
            for i, m in enumerate(self.chi_net):
                mask_ = (species_ == i)
                midx_ = torch.nonzero(mask_, as_tuple=False).flatten()
                if midx_.shape[0] > 0:
                    input_ = input_features.index_select(0, midx_)
                    chi.masked_scatter_(mask_, m(input_).flatten())

            # Qeq
            chi_copy = chi.clone().view_as(species)
            corr = (net_charge + (chi_copy / jii).sum(dim=1)) / (1.0 / jii * padding_mask).sum(dim=1)
            pred_charges = -1.0 / jii * (chi_copy - corr.unsqueeze(-1)) * padding_mask
            assert pred_charges[species == -1].sum() == 0

            esp = ESPerf(pair_padding_mask, distances, pred_charges, species)
            qraev = compute_radial_qaev(
                species,
                coordinates,
                pred_charges,
                self.aev_computer.constants(),
                self.aev_computer.sizes,
                None
            )

        # energy calculation
        input_features = torch.cat(
            (
                aev,
                qraev.flatten(0, 1),
                pred_charges.flatten().unsqueeze(-1),
                esp.flatten().unsqueeze(-1)
            ), dim=1
        )
        for i, m in enumerate(self.ani_nets):
            mask_ = (species_ == i)
            midx_ = torch.nonzero(mask_, as_tuple=False).flatten()
            if midx_.shape[0] > 0:
                input_ = input_features.index_select(0, midx_)
                atomic_energies.masked_scatter_(mask_, m(input_).flatten())

        atomic_energies = atomic_energies.view_as(species)
        mol_energy = torch.sum(atomic_energies, dim=1)

        charge_sq = pred_charges.unsqueeze(1) * pred_charges.unsqueeze(2) * pair_padding_mask
        coulomb = charge_sq * elect_screen(distances) / distances
        elec_energy = torch.triu(coulomb, diagonal=1).sum(dim=(1, 2))

        return species, mol_energy + elec_energy, pred_charges


class JRNN(torch.nn.Module):

    def __init__(self, aev_computer, network_dims, chinet_dims):
        super(JRNN, self).__init__()

        self.aev_computer = aev_computer

        def atomic_net(layers_dims, input_dim):
            assert len(layers_dims) == 3

            return torch.nn.Sequential(
                torch.nn.Linear(input_dim, layers_dims[0]),
                torch.nn.CELU(0.1),
                torch.nn.Linear(layers_dims[0], layers_dims[1]),
                torch.nn.CELU(0.1),
                torch.nn.Linear(layers_dims[1], layers_dims[2]),
                torch.nn.CELU(0.1),
                torch.nn.Linear(layers_dims[2], 1)
            )

        self.ani_nets = torch.nn.ModuleList(
            [
                atomic_net(
                    layers_dims, aev_computer.aev_length + aev_computer.radial_length + 2
                ) for layers_dims in network_dims
            ]
        )
        self.chi_net = torch.nn.Sequential(
            atomic_net(
                chinet_dims,
                aev_computer.aev_length + aev_computer.radial_length + 2
            ),
            torch.nn.Softplus(),
            ActivationHistogram('chi_activation')
        )

    def forward(self, model_input):
        species, coordinates, net_charge = model_input
        _, aev = self.aev_computer((species, coordinates))
        distances = nonzero_batch_dmat(coordinates)

        padding_mask = (species.ne(-1)).float()
        pair_padding_mask = padding_mask.unsqueeze(1) * padding_mask.unsqueeze(2)
        num_atoms = padding_mask.sum(dim=1)

        species_ = species.flatten()
        aev = aev.flatten(0, 1)
        # chi net
        chi = aev.new_zeros(species_.shape)
        pred_charges = aev.new_zeros(species_.shape)
        atomic_energies = aev.new_zeros(species_.shape)
        esp = aev.new_zeros(species_.shape)

        mask = species_.ne(-1)
        midx = torch.nonzero(mask, as_tuple=False).flatten()

        # RNN
        for _ in range(2):
            qraev = compute_radial_qaev(
                species,
                coordinates,
                pred_charges,
                self.aev_computer.constants(),
                self.aev_computer.sizes,
                None
            )

            input_features = torch.cat(
                (
                    aev,
                    qraev.flatten(0, 1),
                    pred_charges.unsqueeze(-1),
                    esp.flatten().unsqueeze(-1)
                ), dim=1
            )
            input_ = input_features.index_select(0, midx)
            chi.masked_scatter_(mask, self.chi_net(input_).flatten())

            # net charge correction
            chi_copy = chi.clone().view_as(species)
            k_net = 1.0 + torch.abs(net_charge) / chi_copy.sum(dim=-1)
            chi_mean = chi_copy.sum(dim=-1) / num_atoms

            k_pmask = net_charge.gt(0)
            k_nmask = net_charge.lt(0)

            k_p = torch.ones_like(k_net)
            k_n = torch.ones_like(k_net)

            k_p.masked_scatter_(k_pmask, k_net[k_pmask])
            k_n.masked_scatter_(k_nmask, k_net[k_nmask])

            # using chisum / chi.sum() to avoid zeros in denom
            iter_charges = - k_n.unsqueeze(-1) * chi_copy + (k_p * chi_mean).unsqueeze(-1)
            iter_charges = iter_charges * padding_mask
            assert iter_charges[species == -1].sum() == 0

            pred_charges = iter_charges.flatten()

            esp = ESPerf(pair_padding_mask, distances, pred_charges.view_as(species), species)

        input_features = torch.cat(
            (
                aev,
                qraev.flatten(0, 1),
                pred_charges.unsqueeze(-1),
                esp.flatten().unsqueeze(-1)
            ), dim=1
        )
        for i, m in enumerate(self.ani_nets):
            mask_ = (species_ == i)
            midx_ = torch.nonzero(mask_, as_tuple=False).flatten()
            if midx_.shape[0] > 0:
                input_ = input_features.index_select(0, midx_)
                atomic_energies.masked_scatter_(mask_, m(input_).flatten())

        atomic_energies = atomic_energies.view_as(species)
        pred_charges = pred_charges.view_as(species)
        mol_energy = torch.sum(atomic_energies, dim=1)
        coulomb_energy = 0.5 * (pred_charges * esp).sum(dim=-1)

        return species, mol_energy + coulomb_energy, pred_charges


class QRNN(torch.nn.Module):
    """
    Charge-only RNN
    """
    def __init__(self, aev_computer, network_dims):
        super(QRNN, self).__init__()

        self.aev_computer = aev_computer

        def atomic_net(layers_dims, input_dim):
            assert len(layers_dims) == 3

            return torch.nn.Sequential(
                torch.nn.Linear(input_dim, layers_dims[0]),
                torch.nn.CELU(0.1),
                torch.nn.Linear(layers_dims[0], layers_dims[1]),
                torch.nn.CELU(0.1),
                torch.nn.Linear(layers_dims[1], layers_dims[2]),
                torch.nn.CELU(0.1),
                torch.nn.Linear(layers_dims[2], 1)
            )

        self.chi_net = torch.nn.Sequential(
            atomic_net(network_dims[0], aev_computer.aev_length + 2),
            torch.nn.Softplus(),
            ActivationHistogram('chi_activation')
        )

    def forward(self, model_input):
        species, coordinates, net_charge = model_input
        _, aev = self.aev_computer((species, coordinates))
        distances = nonzero_batch_dmat(coordinates)

        padding_mask = (species.ne(-1)).float()
        pair_padding_mask = padding_mask.unsqueeze(1) * padding_mask.unsqueeze(2)
        num_atoms = padding_mask.sum(dim=1)

        species_ = species.flatten()
        aev = aev.flatten(0, 1)
        # chi net
        chi = aev.new_zeros(species_.shape)
        pred_charges = aev.new_zeros(species_.shape)
        esp = aev.new_zeros(species_.shape)

        mask = species_.ne(-1)
        midx = torch.nonzero(mask, as_tuple=False).flatten()

        # RNN
        for _ in range(4):
            input_features = torch.cat(
                (
                    aev,
                    pred_charges.unsqueeze(-1),
                    esp.flatten().unsqueeze(-1)
                ), dim=1
            )
            input_ = input_features.index_select(0, midx)
            chi.masked_scatter_(mask, self.chi_net(input_).flatten())

            # net charge correction
            chi_copy = chi.clone().view_as(species)
            k_net = 1.0 + torch.abs(net_charge) / chi_copy.sum(dim=-1)
            chi_mean = chi_copy.sum(dim=-1) / num_atoms

            k_pmask = net_charge.gt(0)
            k_nmask = net_charge.lt(0)

            k_p = torch.ones_like(k_net)
            k_n = torch.ones_like(k_net)

            k_p.masked_scatter_(k_pmask, k_net[k_pmask])
            k_n.masked_scatter_(k_nmask, k_net[k_nmask])

            # using chisum / chi.sum() to avoid zeros in denom
            iter_charges = - k_n.unsqueeze(-1) * chi_copy + (k_p * chi_mean).unsqueeze(-1)
            iter_charges = iter_charges * padding_mask
            assert iter_charges[species == -1].sum() == 0

            pred_charges = iter_charges.flatten()

            esp = ESPerf(pair_padding_mask, distances, pred_charges.view_as(species), species)

        pred_charges = pred_charges.view_as(species)

        return species, pred_charges


class JQSANIModel(torch.nn.Module):

    def __init__(self, aev_computer, network_dims):
        super(JQSANIModel, self).__init__()

        self.aev_computer = aev_computer
        self.esp = 1  # no esp input

        def atomic_net(layers_dims, input_dim):
            assert len(layers_dims) == 3

            return torch.nn.Sequential(
                torch.nn.Linear(input_dim, layers_dims[0]),
                torch.nn.CELU(0.1),
                torch.nn.Linear(layers_dims[0], layers_dims[1]),
                torch.nn.CELU(0.1),
                torch.nn.Linear(layers_dims[1], layers_dims[2]),
                torch.nn.CELU(0.1),
                torch.nn.Linear(layers_dims[2], 1)
            )

        self.ani_nets = torch.nn.ModuleList(
            [
                atomic_net(
                    layers_dims, aev_computer.aev_length + 2 * self.esp
                ) for layers_dims in network_dims
            ]
        )
        self.chi_net = torch.nn.Sequential(
            atomic_net(network_dims[0], aev_computer.aev_length),
            torch.nn.Softplus(),
            ActivationHistogram('chi_activation')
        )

    def forward(self, model_input):
        species, coordinates, net_charge = model_input
        _, aev = self.aev_computer((species, coordinates))
        distances = nonzero_batch_dmat(coordinates)

        padding_mask = (species.ne(-1)).float()
        pair_padding_mask = padding_mask.unsqueeze(1) * padding_mask.unsqueeze(2)
        num_atoms = padding_mask.sum(dim=1)

        species_ = species.flatten()
        aev = aev.flatten(0, 1)

        # chi net
        chi = aev.new_zeros(species_.shape)
        mask = species_.ne(-1)
        midx = torch.nonzero(mask, as_tuple=False).flatten()
        input_ = aev.index_select(0, midx)
        chi.masked_scatter_(mask, self.chi_net(input_).flatten())
        chi = chi.view_as(species)

        # net charge correction
        k_net = 1.0 + torch.abs(net_charge) / chi.sum(dim=-1)
        chi_mean = chi.sum(dim=-1) / num_atoms

        k_pmask = net_charge.gt(0)
        k_nmask = net_charge.lt(0)

        k_p = torch.ones_like(k_net)
        k_n = torch.ones_like(k_net)

        k_p.masked_scatter_(k_pmask, k_net[k_pmask])
        k_n.masked_scatter_(k_nmask, k_net[k_nmask])

        # using chisum / chi.sum() to avoid zeros in denom
        pred_charges = - k_n.unsqueeze(-1) * chi + (k_p * chi_mean).unsqueeze(-1)
        pred_charges = pred_charges * padding_mask
        assert pred_charges[species == -1].sum() == 0

        esp = ESPerf(pair_padding_mask, distances, pred_charges, species)
        # esp = ESP(pair_padding_mask, distances, pred_charges)
        coulomb_energy = 0.5 * (pred_charges * esp).sum(dim=-1)

        # atomic nets
        input_features = (
            aev,
            pred_charges.flatten().unsqueeze(-1),
            esp.flatten().unsqueeze(-1)
        )
        aev = torch.cat(input_features, dim=1)
        output = aev.new_zeros(species_.shape)

        for i, m in enumerate(self.ani_nets):
            mask = (species_ == i)
            midx = torch.nonzero(mask, as_tuple=False).flatten()
            if midx.shape[0] > 0:
                input_ = aev.index_select(0, midx)
                output.masked_scatter_(mask, m(input_).flatten())

        output = output.view_as(species)
        mol_energy = torch.sum(output, dim=1)

        return species, mol_energy + coulomb_energy, pred_charges


class SimpleQSANIModel(torch.nn.Module):

    def __init__(self, aev_computer, network_dims):
        super(SimpleQSANIModel, self).__init__()

        self.aev_computer = aev_computer

        def atomic_net(layers_dims, input_dim):
            assert len(layers_dims) == 3

            return torch.nn.Sequential(
                torch.nn.Linear(input_dim, layers_dims[0]),
                torch.nn.CELU(0.1),
                torch.nn.Linear(layers_dims[0], layers_dims[1]),
                torch.nn.CELU(0.1),
                torch.nn.Linear(layers_dims[1], layers_dims[2]),
                torch.nn.CELU(0.1),
                torch.nn.Linear(layers_dims[2], 1)
            )

        self.ani_nets = torch.nn.ModuleList(
            [
                atomic_net(
                    layers_dims, aev_computer.aev_length + 2
                ) for layers_dims in network_dims
            ]
        )
        self.q_net = torch.nn.Sequential(
            atomic_net(network_dims[0], aev_computer.aev_length),
        )

    def forward(self, model_input):
        species, coordinates, net_charge = model_input
        _, aev = self.aev_computer((species, coordinates))
        distances = nonzero_batch_dmat(coordinates)

        padding_mask = (species.ne(-1)).float()
        pair_padding_mask = padding_mask.unsqueeze(1) * padding_mask.unsqueeze(2)

        species_ = species.flatten()
        aev = aev.flatten(0, 1)

        # charge net
        charges = aev.new_zeros(species_.shape)
        mask = species_.ne(-1)
        midx = torch.nonzero(mask, as_tuple=False).flatten()
        input_ = aev.index_select(0, midx)
        charges.masked_scatter_(mask, self.q_net(input_).flatten())
        charges = charges.view_as(species)

        # net charge correction
        denom = (charges**2).sum(dim=-1).unsqueeze(-1)
        denom[denom < 1e-8] = 1e-8
        abs_ratio = (charges**2) / denom
        pred_charges = charges + (net_charge - charges.sum(dim=-1)).unsqueeze(-1) * abs_ratio

        esp = ESP(pair_padding_mask, distances, pred_charges)
        coulomb_energy = 0.5 * (pred_charges * esp).sum(dim=-1)

        # atomic nets
        input_features = (
            aev,
            pred_charges.flatten().unsqueeze(-1),
            esp.flatten().unsqueeze(-1)
        )
        aev = torch.cat(input_features, dim=1)
        output = aev.new_zeros(species_.shape)

        for i, m in enumerate(self.ani_nets):
            mask = (species_ == i)
            midx = torch.nonzero(mask, as_tuple=False).flatten()
            if midx.shape[0] > 0:
                input_ = aev.index_select(0, midx)
                output.masked_scatter_(mask, m(input_).flatten())

        output = output.view_as(species)
        mol_energy = torch.sum(output, dim=1)

        return species, mol_energy + coulomb_energy, pred_charges


class QModel(torch.nn.Module):

    def __init__(self, aev_computer, network_dims):
        super(QModel, self).__init__()

        self.aev_computer = aev_computer

        def atomic_net(layers_dims, input_dim):
            assert len(layers_dims) == 3

            return torch.nn.Sequential(
                torch.nn.Linear(input_dim, layers_dims[0]),
                torch.nn.CELU(0.1),
                torch.nn.Linear(layers_dims[0], layers_dims[1]),
                torch.nn.CELU(0.1),
                torch.nn.Linear(layers_dims[1], layers_dims[2]),
                torch.nn.CELU(0.1),
                torch.nn.Linear(layers_dims[2], 1)
            )

        self.chi_net = torch.nn.Sequential(
            atomic_net(network_dims[0], aev_computer.aev_length),
            torch.nn.Softplus(),
            ActivationHistogram('chi_activation')
        )

    def forward(self, model_input):
        species, coordinates, net_charge = model_input
        _, aev = self.aev_computer((species, coordinates))
        distances = nonzero_batch_dmat(coordinates)

        padding_mask = (species.ne(-1)).float()
        pair_padding_mask = padding_mask.unsqueeze(1) * padding_mask.unsqueeze(2)

        species_ = species.flatten()
        aev = aev.flatten(0, 1)

        # charge equilibriation
        chi = aev.new_zeros(species_.shape)
        mask = species_.ne(-1)
        midx = torch.nonzero(mask, as_tuple=False).flatten()
        input_ = aev.index_select(0, midx)
        chi.masked_scatter_(mask, self.chi_net(input_).flatten())
        chi = chi.view_as(species)

        coulomb = get_coulomb(pair_padding_mask, species, distances)
        charge_transfered = charge_transfer(
            pair_padding_mask, chi, coulomb, species, distances
        )

        # net charge correction
        denom = (charge_transfered**2).sum(dim=-1).unsqueeze(-1)
        denom[denom < 1e-8] = 1e-8
        abs_ratio = (charge_transfered**2) / denom
        pred_charges = charge_transfered + (net_charge - charge_transfered.sum(dim=-1)).unsqueeze(-1) * abs_ratio

        return species, pred_charges


class QSANIModel(torch.nn.Module):

    def __init__(self, aev_computer, network_dims):
        super(QSANIModel, self).__init__()

        self.aev_computer = aev_computer
        self.esp = 1

        def atomic_net(layers_dims, input_dim):
            assert len(layers_dims) == 3

            return torch.nn.Sequential(
                torch.nn.Linear(input_dim, layers_dims[0]),
                torch.nn.CELU(0.1),
                torch.nn.Linear(layers_dims[0], layers_dims[1]),
                torch.nn.CELU(0.1),
                torch.nn.Linear(layers_dims[1], layers_dims[2]),
                torch.nn.CELU(0.1),
                torch.nn.Linear(layers_dims[2], 1)
            )

        self.ani_nets = torch.nn.ModuleList(
            [
                atomic_net(
                    layers_dims, aev_computer.aev_length + 2 * self.esp
                ) for layers_dims in network_dims
            ]
        )
        self.chi_net = torch.nn.Sequential(
            atomic_net(network_dims[0], aev_computer.aev_length),
            torch.nn.Softplus(),
            ActivationHistogram('chi_activation')
        )

    def equil_energy(self, coulomb, chi, charges):
        """
        Electrostatic energy calculated from charge equilibrium method

        Arguments:
            coulomb matrix (torch.Tensor): (N, n, n)
            chi (torch.Tensor): (N, n)
            charges (torch.Tensor): (N, n)
        Returns:
            energy (torch.Tensor): (N,)
        """
        charge_products = charges.unsqueeze(1) * charges.unsqueeze(2)
        energetic_term = 0.5 * (coulomb * charge_products).sum(dim=(1, 2))
        return energetic_term + (chi * charges).sum(dim=1)

    def forward(self, model_input):
        species, coordinates, net_charge = model_input
        _, aev = self.aev_computer((species, coordinates))
        distances = nonzero_batch_dmat(coordinates)

        padding_mask = (species.ne(-1)).float()
        pair_padding_mask = padding_mask.unsqueeze(1) * padding_mask.unsqueeze(2)

        species_ = species.flatten()
        aev = aev.flatten(0, 1)

        # charge equilibriation
        chi = aev.new_zeros(species_.shape)
        mask = species_.ne(-1)
        midx = torch.nonzero(mask, as_tuple=False).flatten()
        input_ = aev.index_select(0, midx)
        chi.masked_scatter_(mask, self.chi_net(input_).flatten())
        chi = chi.view_as(species)

        coulomb = get_coulomb(pair_padding_mask, species, distances)
        charge_transfered = charge_transfer(
            pair_padding_mask, chi, coulomb, species, distances
        )

        # net charge correction
        denom = (charge_transfered**2).sum(dim=-1).unsqueeze(-1)
        denom[denom < 1e-8] = 1e-8
        abs_ratio = (charge_transfered**2) / denom
        pred_charges = charge_transfered + (net_charge - charge_transfered.sum(dim=-1)).unsqueeze(-1) * abs_ratio
        assert pred_charges[species == -1].sum() == 0

        mol_energy_c = self.equil_energy(coulomb, chi, pred_charges)

        # atomic nets
        if self.esp:
            esp = ESP(pair_padding_mask, distances, pred_charges)
            input_features = (
                aev,
                pred_charges.flatten().unsqueeze(-1),
                esp.flatten().unsqueeze(-1)
            )
            aev = torch.cat(input_features, dim=1)

        output = aev.new_zeros(species_.shape)

        for i, m in enumerate(self.ani_nets):
            mask = (species_ == i)
            midx = torch.nonzero(mask, as_tuple=False).flatten()
            if midx.shape[0] > 0:
                input_ = aev.index_select(0, midx)
                output.masked_scatter_(mask, m(input_).flatten())

        output = output.view_as(species)
        mol_energy = torch.sum(output, dim=1)

        return species, mol_energy + mol_energy_c, pred_charges


class SharedSANIModel(torch.nn.Module):

    def __init__(self, aev_computer, network_dims, out_dim):
        super(SharedSANIModel, self).__init__()

        self.aev_computer = aev_computer
        self.out_dim = out_dim
        shared_net_dims = [2 * out_dim + 4, 32, 16, 1]

        def atomic_net(layers_dim):
            assert len(layers_dim) == 3
            return torch.nn.Sequential(
                torch.nn.Linear(aev_computer.aev_length, layers_dim[0]),
                torch.nn.CELU(0.1),
                torch.nn.Linear(layers_dim[0], layers_dim[1]),
                torch.nn.CELU(0.1),
                torch.nn.Linear(layers_dim[1], layers_dim[2]),
                torch.nn.CELU(0.1),
                torch.nn.Linear(layers_dim[2], self.out_dim)
            )

        self.ani_nets = torch.nn.ModuleList(
            [atomic_net(layers_dim) for layers_dim in network_dims]
        )
        self.shared_net = torch.nn.Sequential(
                ActivationHistogram('mol_input_activation'),
                torch.nn.Linear(shared_net_dims[0], shared_net_dims[1]),
                torch.nn.CELU(1.0),
                torch.nn.Linear(shared_net_dims[1], shared_net_dims[2]),
                torch.nn.CELU(1.0),
                torch.nn.Linear(shared_net_dims[2], shared_net_dims[3])
            )

    def dfrom_centroid(self, mask, coordinates):
        assert len(coordinates.size()) == 3
        num_atoms = mask.sum(dim=1)
        centeriod = coordinates.sum(dim=1) / num_atoms.unsqueeze(-1)  # N, 3
        diff_sq = (coordinates - centeriod.unsqueeze(1)) ** 2
        diff_sq = diff_sq * mask.unsqueeze(-1)
        return diff_sq.sum(dim=-1).sqrt()  # N, n (with padding)

    def get_mol_features(
        self, species, atomic_output, coordinates=None, net_charge=None
    ):
        mask = (species.ne(-1)).to(atomic_output.dtype).unsqueeze(-1)  # N,n,1
        num_atoms = mask.sum(dim=1)  # N,1

        sum_ = torch.sum(atomic_output, dim=1)  # N,self.out_dim
        mean_ = sum_ / num_atoms

        # calculate stddev of padded tensor
        # producing nan in grads!
        # diff = (atomic_output - mean_.unsqueeze(-1)) ** 2
        # sum_diff = torch.sum(diff * mask, dim=1)
        # std_ = torch.sqrt(sum_diff / num_atoms)

        # smooth min and max
        # avoided using torch.logsumexp to use vectorized OP
        # TODO: vectorized way of min, max for padded tensor
        # min_ = []
        # max_ = []

        # for mol_idx in range(atomic_output.shape[0]):
        #     n = int(num_atoms[mol_idx])
        #     nonpaded = atomic_output[mol_idx][:n][:]
        #     min_.append(torch.min(nonpaded, dim=0)[0])
        #     max_.append(torch.max(nonpaded, dim=0)[0])

        # min_ = torch.stack(min_, dim=0)  # N,self.out_dim
        # max_ = torch.stack(max_, dim=0)  # N,self.out_dim

        # smoothmax_ = torch.log(
        #     torch.sum(
        #         torch.exp(atomic_output - max_.unsqueeze(1)) * mask,
        #         dim=1
        #     )
        # ) + max_

        # smoothmin_ = - torch.log(
        #     torch.sum(
        #         torch.exp(- atomic_output + min_.unsqueeze(1)) * mask,
        #         dim=1
        #     )
        # ) + min_

        ret = (sum_, mean_)

        # calculate distance from centroid
        if torch.is_tensor(coordinates):
            mask = mask.squeeze()
            atom_dist_centroid = self.dfrom_centroid(mask, coordinates)  # N,n

            sum_dist = torch.sum(atom_dist_centroid, dim=1)
            mean_dist = sum_dist / num_atoms.squeeze()
            max_dist = torch.max(atom_dist_centroid, dim=1)[0]
            smoothmax_dist = torch.log(
                torch.sum(
                    torch.exp(atom_dist_centroid - max_dist.unsqueeze(-1)) * mask,
                    dim=1
                )
            ) + max_dist

            ret = ret + (
                sum_dist.unsqueeze(-1),
                mean_dist.unsqueeze(-1),
                smoothmax_dist.unsqueeze(-1)
            )

        # add net charge as a feature
        if torch.is_tensor(net_charge):
            ret = ret + (net_charge.unsqueeze(-1),)

        return ret

    def forward(self, species_coordinates):
        species, coordinates, net_charge = species_coordinates
        _, aev = self.aev_computer((species, coordinates))
        species_ = species.flatten()
        aev = aev.flatten(0, 1)

        output = aev.new_zeros(*species_.shape, self.out_dim)

        for i, m in enumerate(self.ani_nets):
            mask = (species_ == i)
            midx = torch.nonzero(mask, as_tuple=False).flatten()
            if midx.shape[0] > 0:
                input_ = aev.index_select(0, midx)
                output.masked_scatter_(mask.unsqueeze(-1), m(input_))

        output = output.view(*species.size(), -1)  # N, n, self.out_dim

        # molecular properties
        mol_features = torch.cat(
            self.get_mol_features(species, output, coordinates, net_charge),
            dim=1
        )

        mol_energies = self.shared_net(mol_features).squeeze(-1)
        return species, mol_energies


class SANIModel(torch.nn.Module):
    """SANI model that compute energies from species and AEVs.

    Different atom types might have different modules, when computing
    energies, for each atom, the module for its corresponding atom type will
    be applied to its AEV, after that, outputs of modules will be reduced along
    different atoms to obtain molecular energies.

    .. warning::

        The species must be indexed in 0, 1, 2, 3, ..., not the element
        index in periodic table. Check :class:`torchani.SpeciesConverter`
        if you want periodic table indexing.

    .. note:: The resulting energies are in Hartree.

    Arguments:
        modules (:class:`collections.abc.Sequence`): Modules for each atom
            types. Atom types are distinguished by their order in
            :attr:`modules`, which means, for example ``modules[i]`` must be
            the module for atom type ``i``. Different atom types can share a
            module by putting the same reference in :attr:`modules`.
    """

    def __init__(self, aev_computer, network_dims):
        super(SANIModel, self).__init__()

        self.aev_computer = aev_computer

        def atomic_net(layers_dim):
            assert len(layers_dim) == 3
            return torch.nn.Sequential(
                torch.nn.Linear(aev_computer.aev_length, layers_dim[0]),
                torch.nn.CELU(0.1),
                torch.nn.Linear(layers_dim[0], layers_dim[1]),
                torch.nn.CELU(0.1),
                torch.nn.Linear(layers_dim[1], layers_dim[2]),
                torch.nn.CELU(0.1),
                torch.nn.Linear(layers_dim[2], 1)
            )

        self.ani_nets = torch.nn.ModuleList(
            [atomic_net(layers_dim) for layers_dim in network_dims]
        )

    def forward(self, species_coordinates):
        species, aev = self.aev_computer(species_coordinates)
        species_ = species.flatten()
        aev = aev.flatten(0, 1)

        output = aev.new_zeros(species_.shape)

        for i, m in enumerate(self.ani_nets):
            mask = (species_ == i)
            midx = torch.nonzero(mask, as_tuple=False).flatten()
            if midx.shape[0] > 0:
                input_ = aev.index_select(0, midx)
                output.masked_scatter_(mask, m(input_).flatten())
        output = output.view_as(species)
        return species, torch.sum(output, dim=1)


class EMA:
    """Exponential moving average of model parameters.
    https://anmoljoshi.com/Pytorch-Dicussions/
    Arguments:
        model (torch.nn.Module): Model with parameters whose EMA will be kept.
        decay (float): Decay rate for exponential moving average.
    """
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        self.original = {}

        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data

    def __call__(self, model):
        decay = self.decay
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average

    def assign(self, model):
        """Assign exponential moving average of parameter values to the
        respective parameters.
        Arguments:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data
                param.data = self.shadow[name]

    def resume(self, model):
        """Restore original parameters to a model. That is, put back
        the values that were in each parameter at the last call to `assign`.
        Arguments:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]


class MTLLoss(torch.nn.Module):
    """Args:
            losses: a list of task specific loss terms
            num_tasks: number of tasks
    """

    def __init__(self, num_tasks=2):
        super(MTLLoss, self).__init__()
        self.num_tasks = num_tasks
        self.log_sigma = torch.nn.Parameter(torch.zeros((num_tasks)))

    def get_precisions(self):
        return 0.5 * torch.exp(- self.log_sigma) ** 2

    def forward(self, *loss_terms):
        assert len(loss_terms) == self.num_tasks

        total_loss = 0
        self.precisions = self.get_precisions()

        for task in range(self.num_tasks):
            total_loss += self.precisions[task] * loss_terms[task] + self.log_sigma[task]

        return total_loss
