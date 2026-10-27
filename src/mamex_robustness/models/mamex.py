import torch
import torch.nn.functional as F

from mamex_robustness.domain.gating import cross_modal_cosine, vgr_reweight

class MoE_Layer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # List of expert networks
        self.experts = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, output_dim) for _ in range(num_experts)
        ])
        
        # Gate network that combines the experts
        self.gate = torch.nn.Linear(input_dim, num_experts)

    def forward(self, x):
        # Output of each expert: [batch_size, num_experts, output_dim]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        
        # Gate logits: [batch_size, num_experts]
        gate_logits = self.gate(x)
        # Auxiliary loss for load balancing
        aux_loss = self._compute_load_balancing_loss(gate_logits)

        # Select the top_k experts based on the logits
        topk_logits, topk_indices = torch.topk(gate_logits, self.top_k, dim=1)
        
        # Apply softmax only to the selected top_k logits
        topk_weights = F.softmax(topk_logits, dim=1)  # [batch_size, top_k]
        
        # Select the outputs of the top_k experts
        selected_experts = expert_outputs.gather(1, topk_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.size(-1)))
        
        # Weighted sum of the outputs of the top_k experts
        topk_weights = topk_weights.unsqueeze(-1)  # [batch_size, top_k, 1]
        output = torch.sum(topk_weights * selected_experts, dim=1)  # [batch_size, output_dim]
        
        return output, aux_loss
    def _compute_load_balancing_loss(self, gate_logits):
        """
        Compute the auxiliary loss that balances the load across experts
        Uses two components:
        1. Importance loss: ensures every expert is used about equally
        2. Load loss: avoids overloading an expert
        """
        # Probability of each expert
        gates = F.softmax(gate_logits, dim=-1)  # [batch_size, num_experts]
        
        importance_per_expert = gates.mean(dim=0)  # [num_experts]
        # The target is a uniform distribution over experts (1/num_experts)
        target_importance = torch.ones_like(importance_per_expert) / self.num_experts
        importance_loss = F.kl_div(
                importance_per_expert.log(),
                target_importance,
                reduction='sum'
            )
        
        return importance_loss


class MOME_model(torch.nn.Module):
    def __init__(self, env, dataset):
        super(MOME_model, self).__init__()
        self.env = env
        self.n_user = dataset.n_user
        self.m_item = dataset.m_item
        self.free_emb_dimension = self.env.args.free_emb_dimension
        
        # Process the input features
        self.feature = torch.tensor(dataset.feature, dtype=torch.float32).to(self.env.device)
        self.feature = torch.nn.functional.normalize(self.feature)
        self.image_feat = torch.tensor(dataset.image_feat, dtype=torch.float32).to(self.env.device)
        self.image_feat = torch.nn.functional.normalize(self.image_feat)
        self.text_feat = torch.tensor(dataset.text_feat, dtype=torch.float32).to(self.env.device)
        self.text_feat = torch.nn.functional.normalize(self.text_feat)

        # User embeddings
        self.user_emb = torch.nn.Embedding(self.n_user, self.free_emb_dimension)
        
        # MoE layers
        self.image_linear = MoE_Layer(self.image_feat.shape[1], self.free_emb_dimension, 4)
        self.text_linear = MoE_Layer(self.text_feat.shape[1], self.free_emb_dimension, 4)
        # Gating network
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(2 * self.free_emb_dimension, self.free_emb_dimension),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(self.free_emb_dimension, 2)
        )
        
        # Visual Grounding Regularization (parameter-free; can be toggled on a trained model)
        self.use_vgr = bool(getattr(self.env.args, 'use_vgr', 0))
        self.vgr_gamma = float(getattr(self.env.args, 'vgr_gamma', 8.0))
        self.vgr_control = getattr(self.env.args, 'vgr_control', 'cosine')
        self.last_gate_weights = None
        self.last_cos = None
        self._eval_text = None

        # Initialize the weights
        torch.nn.init.normal_(self.user_emb.weight, std=0.1)
        self.to(self.env.device)

    def set_eval_text(self, text_feat_noisy):
        """Use these (e.g. corrupted) text features in forward until cleared with None; self.text_feat is kept."""
        if text_feat_noisy is None:
            self._eval_text = None
            return
        text = torch.tensor(text_feat_noisy, dtype=torch.float32).to(self.env.device)
        self._eval_text = torch.nn.functional.normalize(text)

    def forward(self, random=False):
        user_emb = self.user_emb.weight
        text_feat = self.text_feat if self._eval_text is None else self._eval_text

        # MoE embeddings
        image_emb, image_aux = self.image_linear(self.image_feat)
        text_emb, text_aux = self.text_linear(text_feat)
        
        # Concatenate and compute the gate
        combined_emb = torch.cat([image_emb, text_emb], dim=1)
        gate_outputs = self.gate(combined_emb)

        # Compute the gate weights
        gate_weights = torch.softmax(gate_outputs, dim=1)
        
        fusion_weights = gate_weights
        if self.use_vgr:
            fusion_weights = vgr_reweight(gate_weights, image_emb, text_emb, self.vgr_gamma, self.vgr_control)
        self.last_gate_weights = fusion_weights.detach()
        self.last_cos = cross_modal_cosine(image_emb, text_emb)

        # Weighted sum of the modality embeddings
        item_emb = (
            fusion_weights[:, 0].unsqueeze(1) * image_emb +
            fusion_weights[:, 1].unsqueeze(1) * text_emb
        )
        
        # Add balance loss to encourage equal usage of experts
        mean_gate_weights = gate_weights.mean(dim=0)  # [2]
        target_weights = torch.ones_like(mean_gate_weights) / mean_gate_weights.size(0)
        balance_loss = torch.nn.functional.kl_div(
            mean_gate_weights.log(),
            target_weights,
            reduction='sum'
        )
        return user_emb, item_emb, image_emb, text_emb, image_aux + text_aux, balance_loss