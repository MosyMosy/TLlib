import torch
from torch import nn


class NearestPrior(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Feature_all, logit_all, y_source, device):
        source_size = torch.numel(y_source)
        Feature_source = Feature_all[:source_size]
        Feature_target = Feature_all[source_size:]

        logit_source = logit_all[:source_size]
        logit_target = logit_all[source_size:]

        loss_cl = nn.CrossEntropyLoss()(logit_source, y_source)
        loss_ce = NearestPrior.entropy(logit_target.softmax(1), dim=1)
        # loss_ce = ce_loss(logit_target, Feature_target)
        loss_reg = NearestPrior.reg_loss_dist(Feature_all= Feature_all, source_size= source_size, device=device)

        return loss_cl, loss_ce, loss_reg


    @staticmethod
    def entropy(p, dim=1):
        return torch.sum(-p * torch.log2(torch.clamp(p, min=1e-8)), dim=dim).mean()

    @staticmethod
    def reg_loss_entropy(Feature_all, source_size, device, sigma=1):
        all_size = Feature_all.size()[0]
        dist_all = torch.cdist(Feature_all, Feature_all, p=2,
                               compute_mode='use_mm_for_euclid_dist_if_necessary').to(device)

        # variances for Gaussian similarity kernel
        k = min(5, min(source_size, (all_size - source_size)))
        var_all = torch.zeros(all_size, all_size)
        var_all[:source_size, :source_size] = torch.sqrt(torch.max(torch.topk(
            dist_all[:source_size, :source_size], k=k, dim=1, largest=False).values, dim=1)[0] + 1e-8)  # source_intra_var
        var_all[:source_size, source_size:] = torch.sqrt(torch.max(torch.topk(
            dist_all[:source_size, source_size:], k=k, dim=1, largest=False).values, dim=1)[0] + 1e-8)  # source_inter_var
        var_all[source_size:, source_size:] = torch.sqrt(torch.max(torch.topk(
            dist_all[source_size:, source_size:], k=k, dim=1, largest=False).values, dim=1)[0] + 1e-8)  # target_intra_var
        var_all[source_size:, :source_size] = torch.sqrt(torch.max(torch.topk(
            dist_all[source_size:, :source_size], k=k, dim=1, largest=False).values, dim=1)[0] + 1e-8)  # target_inter_var
        var_all = var_all.to(device)
        # Gaussian similarity kernel
        similarity_all = torch.exp(-dist_all / (
            2 * (var_all * sigma) ** 2)) * (1 - torch.eye(all_size).to(device))

        source_intra_similarity_max = torch.max(
            similarity_all[:source_size, :source_size], dim=1)[0]
        source_inter_similarity_max = torch.max(
            similarity_all[:source_size, source_size:], dim=1)[0]
        target_intra_similarity_max = torch.max(
            similarity_all[source_size:, source_size:], dim=1)[0]
        target_inter_similarity_max = torch.max(
            similarity_all[source_size:, :source_size], dim=1)[0]

        source_prob = source_intra_similarity_max / \
            (source_intra_similarity_max + source_inter_similarity_max)
        target_prob = target_intra_similarity_max / \
            (target_intra_similarity_max + target_inter_similarity_max)
        source_probs = torch.stack([source_prob, 1 - source_prob], dim=1).to(device)
        target_probs = torch.stack([target_prob, 1 - target_prob], dim=1).to(device)

        max_entropy = torch.log2(torch.tensor(2)).to(device) * 2
        return (max_entropy - (NearestPrior.entropy(source_probs) + NearestPrior.entropy(target_probs))).to(device)

    @staticmethod
    def reg_loss_dist(Feature_all, source_size, device, sigma=1):
        all_size = Feature_all.size()[0]
        dist_all = torch.cdist(Feature_all, Feature_all, p=2,
                               compute_mode='use_mm_for_euclid_dist_if_necessary').to(device)

        # variances for Gaussian similarity kernel
        k = min(5, min(source_size, (all_size - source_size)))
        var_all = torch.zeros(all_size, all_size)
        var_all[:source_size, :source_size] = torch.sqrt(torch.max(torch.topk(
            dist_all[:source_size, :source_size], k=k, dim=1, largest=False).values, dim=1)[0] + 1e-8)  # source_intra_var
        var_all[:source_size, source_size:] = torch.sqrt(torch.max(torch.topk(
            dist_all[:source_size, source_size:], k=k, dim=1, largest=False).values, dim=1)[0] + 1e-8)  # source_inter_var
        var_all[source_size:, source_size:] = torch.sqrt(torch.max(torch.topk(
            dist_all[source_size:, source_size:], k=k, dim=1, largest=False).values, dim=1)[0] + 1e-8)  # target_intra_var
        var_all[source_size:, :source_size] = torch.sqrt(torch.max(torch.topk(
            dist_all[source_size:, :source_size], k=k, dim=1, largest=False).values, dim=1)[0] + 1e-8)  # target_inter_var
        var_all = var_all.to(device)
        
        # Gaussian similarity kernel
        diagonal = 1 - torch.eye(all_size).to(device)
        similarity_all = torch.exp(-dist_all / (
            2 * (var_all * sigma) ** 2)) * diagonal

        source_intra_similarity_max = torch.max(
            similarity_all[:source_size, :source_size], dim=1)[0]
        source_inter_similarity_max = torch.max(
            similarity_all[:source_size, source_size:], dim=1)[0]
        target_intra_similarity_max = torch.max(
            similarity_all[source_size:, source_size:], dim=1)[0]
        target_inter_similarity_max = torch.max(
            similarity_all[source_size:, :source_size], dim=1)[0]

        source_samples_dist = torch.abs(
            source_intra_similarity_max - source_inter_similarity_max).mean()
        target_samples_dist = torch.abs(
            target_intra_similarity_max - target_inter_similarity_max).mean()

        return (source_samples_dist.to(device) + target_samples_dist.to(device))

    