import torch
from torch import nn


class NearestPrior(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Feature_all, logit_all, y_source, device, k=1):
        source_size = torch.numel(y_source)
        Feature_source = Feature_all[:source_size]
        Feature_target = Feature_all[source_size:]

        logit_source = logit_all[:source_size]
        logit_target = logit_all[source_size:]

        loss_cl = nn.CrossEntropyLoss()(logit_source, y_source)
        loss_ce = NearestPrior.entropy(logit_target.softmax(1), dim=1)
        # loss_ce = ce_loss(logit_target, Feature_target)
        loss_reg = NearestPrior.reg_loss_entropy(Feature_all= Feature_all, source_size= source_size, device=device, k=k)

        return loss_cl, loss_ce, loss_reg


    @staticmethod
    def entropy(p, dim=1):
        return torch.sum(-p * torch.log2(torch.clamp(p, min=1e-8)), dim=dim).mean()

    @staticmethod
    def reg_loss_entropy(Feature_all, source_size, device, k=1):
        similarity_all = NearestPrior.gaussian_similarity_kernel(Feature_all, source_size, device, k)  
        
        source_intra_similarity_max = torch.max(
            similarity_all[:source_size, :source_size], dim=1)[0].detach()
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
    def gaussian_similarity_kernel(Feature_all, source_size, device, k=1):
        all_size = Feature_all.size()[0]
        dist_all = torch.cdist(Feature_all, Feature_all, p=2,
                            compute_mode='use_mm_for_euclid_dist_if_necessary').to(device)
        dist_all = dist_all + ((dist_all.max().detach() + 1) * torch.eye(all_size))
        
        # variances for Gaussian similarity kernel
        k = min(5, min(source_size, (all_size - source_size)))
        var_all = torch.zeros(all_size, all_size)
        var_all[:source_size, :source_size] = torch.clamp(torch.max(torch.topk(
            dist_all[:source_size, :source_size], k=k, dim=1, largest=False).values, dim=1)[0], min=1e-8)  # source_intra_var
        var_all[source_size:, :source_size] = torch.clamp(torch.max(torch.topk(
            dist_all[:source_size, source_size:], k=k, dim=1, largest=False).values, dim=1)[0], min=1e-8)  # source_inter_var
        var_all[source_size:, source_size:] = torch.clamp(torch.max(torch.topk(
            dist_all[source_size:, source_size:], k=k, dim=1, largest=False).values, dim=1)[0], min=1e-8)  # target_intra_var
        var_all[:source_size, source_size:] = torch.clamp(torch.max(torch.topk(
            dist_all[source_size:, :source_size], k=k, dim=1, largest=False).values, dim=1)[0], min=1e-8)  # target_inter_var
        var_all = torch.transpose(var_all, dim0=0, dim1=1)
        var_all = var_all.to(device)
        # Gaussian similarity kernel
        similarity_all = torch.exp(-dist_all / (
            2 * (var_all) ** 2)) * (1 - torch.eye(all_size).to(device))
        return similarity_all
    
