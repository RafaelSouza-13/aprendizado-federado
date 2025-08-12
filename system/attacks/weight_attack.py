import torch

class WeightAttack:
    def __init__(self):
        pass

    def zero_weight_attack(self, model):
        return {k: torch.zeros_like(v) for k, v in model.items()}

    def random_weight_attack(self, model):
        return {k: torch.randn_like(v) for k, v in model.items()}

    def permute_weight_attack(self, model):
        poisoned = {}
        for k, v in model.items():
            flat = v.flatten()
            permuted_indices = torch.randperm(flat.numel())
            shuffled = flat[permuted_indices].reshape(v.shape)
            poisoned[k] = shuffled
        return poisoned

        