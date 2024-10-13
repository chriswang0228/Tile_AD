import torch
import torch.nn.functional as func


# vanilla_logits := logits w/o rpl_ block
# logits := logits w/ rpl_ block
def disimilarity_entropy(logits, vanilla_logits,t=1.):
    n_prob = torch.clamp(torch.softmax(vanilla_logits, dim=1), min=1e-7)
    a_prob = torch.clamp(torch.softmax(logits, dim=1), min=1e-7)

    n_entropy = -torch.sum(n_prob * torch.log(n_prob), dim=1) / t
    a_entropy = -torch.sum(a_prob * torch.log(a_prob), dim=1) / t

    entropy_disimilarity = torch.nn.functional.mse_loss(input=a_entropy, target=n_entropy, reduction="none")

    assert ~torch.isnan(entropy_disimilarity).any(), print(torch.min(n_entropy), torch.max(a_entropy))

    return entropy_disimilarity


def energy_loss(logits, targets, vanilla_logits, out_idx=254, t=1.):
    out_msk = (targets == out_idx)
    void_msk = (targets == 255)

    pseudo_targets = torch.argmax(vanilla_logits, dim=1)

    outlier_msk = (out_msk | void_msk)
    energy = -(t * torch.logsumexp(logits / t, dim=1))
    assert ~torch.isnan(energy).any(), "nan check 1"

    entropy_part = func.cross_entropy(input=logits, target=pseudo_targets, reduction='none')[~outlier_msk]

    assert ~torch.isnan(entropy_part).any(), "nan check 2"

    reg = disimilarity_entropy(logits=logits, vanilla_logits=vanilla_logits)[~outlier_msk]

    if torch.sum(out_msk) > 0:
        logits = logits.flatten(start_dim=2).permute(0, 2, 1)
        energy_part = torch.nn.functional.relu(torch.log(torch.sum(torch.exp(logits),
                                                                   dim=2))[out_msk.flatten(start_dim=1)]).mean()
    else:
        energy_part = torch.tensor([.0], device=targets.device)
    return {"entropy_part": entropy_part.mean(), "reg": reg.mean(), "energy_part": energy_part}

def energy_loss_3d(logits, targets, out_idx=254):
    out_msk = (targets == out_idx)
    void_msk = (targets == 255)

    outlier_msk = (out_msk | void_msk)
    ano_map = -(torch.logsumexp(logits, dim=1))
    assert ~torch.isnan(ano_map).any(), "nan check 1"
    entropy_part = func.cross_entropy(input=logits, target=targets, ignore_index=out_idx, reduction='none')[~outlier_msk]
    assert ~torch.isnan(entropy_part).any(), "nan check 2"

    if torch.sum(out_msk) > 0:
        logits = logits.flatten(start_dim=2).permute(0, 2, 1)
        energy_part = torch.nn.functional.relu(torch.log(torch.sum(torch.exp(logits),
                                                                   dim=2))[out_msk.flatten(start_dim=1)]).mean()
    else:
        energy_part = torch.tensor([.0], device=targets.device)
            
    return {"entropy_part": entropy_part.mean(), "energy_part": energy_part}

def cl_loss(logits, targets, out_idx=254):
    outlier_msk = (targets == out_idx)
    score = -torch.max(logits, dim = 1)[0]  
    ood_score = score[outlier_msk]
    id_score = score[~outlier_msk]
    loss = torch.pow(id_score, 2).mean()
    if outlier_msk.sum() > 0:
        loss = loss + torch.pow(torch.clamp(1.0 - ood_score, min=0.0), 2).mean()
    
    return loss

def pixel_loss(logits, targets, outlier_loss='energy', out_idx=254, source=True):
    out_msk = (targets == out_idx)
    void_msk = (targets == 255)

    outlier_msk = (out_msk | void_msk)
    ano_map = -(torch.logsumexp(logits, dim=1))
    assert ~torch.isnan(ano_map).any(), "nan check 1"
    if source:
        entropy_part = func.cross_entropy(input=logits, target=targets, ignore_index=out_idx, reduction='none')[~outlier_msk]
        assert ~torch.isnan(entropy_part).any(), "nan check 2"
    else:
        entropy_part = torch.tensor([.0], device=targets.device)
        
    if outlier_loss=='energy':
        score = -torch.logsumexp(logits, dim=1)
    elif outlier_loss == "rba":
        score = logits.tanh()
        score = -score.sum(dim=1)
    ood_score = score[outlier_msk]
    id_score = score[~outlier_msk]
    loss = torch.pow(torch.nn.functional.relu(id_score - 0.0), 2).mean()
    if outlier_msk.sum() > 0:
        loss = loss + torch.pow(torch.nn.functional.relu(5.0 - ood_score), 2).mean()
        loss = 0.5 * loss    
    return {"entropy_part": entropy_part.mean(), "energy_part": loss}

class PairWiseLoss(torch.nn.Module):
    def __init__(self):
        super(PairWiseLoss, self).__init__()
        
    def forward(self, feats_S, feats_T):
        loss = 0.0
        for i in range(len(feats_S)):
            feat_S = feats_S[i]
            feat_T = feats_T[i]
            loss += func.cross_entropy(input=feat_S, target=feat_T, reduction='none')

        return loss
    
    def L2(self, f_):
        return (((f_**2).sum(dim=1))**0.5).reshape(f_.shape[0],1,f_.shape[2],f_.shape[3]) + 1e-8

    def similarity(self, feat):
        feat = feat.float()
        tmp = self.L2(feat).detach()
        feat = feat/tmp
        feat = feat.reshape(feat.shape[0],feat.shape[1],-1)
        return torch.einsum('icm,icn->imn', [feat, feat])

    def sim_dis_compute(self, f_S, f_T):
        sim_err = ((self.similarity(f_T) - self.similarity(f_S))**2)/((f_T.shape[-1]*f_T.shape[-2])**2)/f_T.shape[0]
        sim_dis = sim_err.sum()
        return sim_dis