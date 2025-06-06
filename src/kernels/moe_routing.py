"""Mixture of Experts (MoE) Routing Implementation

Efficient routing and load balancing for MoE architectures with
hardware-optimized expert selection and token dispatching.

References:
- Switch Transformer: Scaling to Trillion Parameter Models
- GLaM: Efficient Scaling of Language Models with Mixture-of-Experts
- ST-MoE: Designing Stable and Transferable Sparse Expert Models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Tuple, Optional, Dict, Any
import math

from .base_kernel import BaseKernel, KernelMetrics


@triton.jit
def top_k_gating_kernel(
    logits,  # [batch_size * seq_len, num_experts]
    gates,   # [batch_size * seq_len, top_k]
    indices, # [batch_size * seq_len, top_k]
    num_tokens: tl.constexpr,
    num_experts: tl.constexpr,
    top_k: tl.constexpr,
):
    """Triton kernel for efficient top-k expert selection"""
    
    token_idx = tl.program_id(0)
    if token_idx >= num_tokens:
        return
    
    # Load logits for this token
    logits_ptr = logits + token_idx * num_experts
    logits_vals = tl.load(logits_ptr + tl.arange(0, num_experts))
    
    # Apply softmax
    logits_max = tl.max(logits_vals)
    logits_shifted = logits_vals - logits_max
    logits_exp = tl.exp(logits_shifted)
    logits_sum = tl.sum(logits_exp)
    probs = logits_exp / logits_sum
    
    # Find top-k experts (simplified - in practice would need sorting)
    # For demonstration, we'll take the first k elements
    gates_ptr = gates + token_idx * top_k
    indices_ptr = indices + token_idx * top_k
    
    for k in range(top_k):
        tl.store(gates_ptr + k, probs[k])
        tl.store(indices_ptr + k, k)


@triton.jit
def expert_dispatch_kernel(
    tokens,     # [num_tokens, d_model]
    expert_tokens, # [num_experts, max_tokens_per_expert, d_model]
    gates,      # [num_tokens, top_k]
    indices,    # [num_tokens, top_k]
    token_counts, # [num_experts]
    num_tokens: tl.constexpr,
    d_model: tl.constexpr,
    top_k: tl.constexpr,
    max_tokens_per_expert: tl.constexpr,
):
    """Kernel for dispatching tokens to experts"""
    
    token_idx = tl.program_id(0)
    if token_idx >= num_tokens:
        return
    
    # Load token data
    token_ptr = tokens + token_idx * d_model
    token_data = tl.load(token_ptr + tl.arange(0, d_model))
    
    # Process each of the top-k experts for this token
    for k in range(top_k):
        expert_idx = tl.load(indices + token_idx * top_k + k)
        gate_val = tl.load(gates + token_idx * top_k + k)
        
        # Get current token count for this expert
        current_count = tl.atomic_add(token_counts + expert_idx, 1)
        
        if current_count < max_tokens_per_expert:
            # Store token in expert's buffer
            expert_ptr = (expert_tokens + 
                         expert_idx * max_tokens_per_expert * d_model +
                         current_count * d_model)
            
            # Scale token by gate value
            scaled_token = token_data * gate_val
            tl.store(expert_ptr + tl.arange(0, d_model), scaled_token)


class MoERouter(BaseKernel):
    """Optimized MoE routing with load balancing"""
    
    def __init__(self, num_experts: int, top_k: int = 2, 
                 device: str = "cuda", load_balance_weight: float = 0.1):
        super().__init__(device)
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight
        
        # Expert capacity for load balancing
        self.expert_capacity_factor = 1.25
        
    def forward(self, tokens: torch.Tensor, 
                gating_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Route tokens to experts based on gating logits
        
        Args:
            tokens: [batch_size, seq_len, d_model] input tokens
            gating_logits: [batch_size, seq_len, num_experts] expert scores
            
        Returns:
            expert_outputs: Processed tokens from experts
            combine_weights: Weights for combining expert outputs
            aux_loss_info: Information for auxiliary losses
        """
        batch_size, seq_len, d_model = tokens.shape
        num_tokens = batch_size * seq_len
        
        # Reshape for processing
        tokens_flat = tokens.reshape(num_tokens, d_model)
        gating_logits_flat = gating_logits.reshape(num_tokens, self.num_experts)
        
        # Compute expert selection and gates
        gates, expert_indices = self._compute_gating(gating_logits_flat)
        
        # Dispatch tokens to experts
        expert_outputs, token_counts = self._dispatch_tokens(
            tokens_flat, gates, expert_indices
        )
        
        # Combine expert outputs
        combined_output = self._combine_expert_outputs(
            expert_outputs, gates, expert_indices, num_tokens, d_model
        )
        
        # Compute auxiliary losses for load balancing
        aux_loss_info = self._compute_auxiliary_losses(
            gating_logits_flat, expert_indices, token_counts
        )
        
        # Reshape back to original dimensions
        combined_output = combined_output.reshape(batch_size, seq_len, d_model)
        
        return combined_output, gates, aux_loss_info
    
    def _compute_gating(self, gating_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute expert gates and indices using top-k selection"""
        
        # Apply softmax to gating logits
        gates_softmax = F.softmax(gating_logits, dim=-1)
        
        # Select top-k experts
        top_k_gates, top_k_indices = torch.topk(
            gates_softmax, self.top_k, dim=-1
        )
        
        # Renormalize selected gates
        top_k_gates = top_k_gates / top_k_gates.sum(dim=-1, keepdim=True)
        
        return top_k_gates, top_k_indices
    
    def _dispatch_tokens(self, tokens: torch.Tensor, gates: torch.Tensor,
                        expert_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dispatch tokens to their assigned experts"""
        num_tokens, d_model = tokens.shape
        
        # Calculate expert capacity
        expert_capacity = int(
            self.expert_capacity_factor * num_tokens * self.top_k / self.num_experts
        )
        
        # Initialize expert token storage
        expert_tokens = torch.zeros(
            self.num_experts, expert_capacity, d_model,
            device=tokens.device, dtype=tokens.dtype
        )
        
        # Track token counts per expert
        token_counts = torch.zeros(self.num_experts, device=tokens.device, dtype=torch.long)
        
        # Process each token
        for token_idx in range(num_tokens):
            token = tokens[token_idx]
            
            for k in range(self.top_k):
                expert_idx = expert_indices[token_idx, k].item()
                gate_val = gates[token_idx, k]
                
                current_count = token_counts[expert_idx].item()
                
                if current_count < expert_capacity:
                    # Store scaled token
                    expert_tokens[expert_idx, current_count] = token * gate_val
                    token_counts[expert_idx] += 1
        
        return expert_tokens, token_counts
    
    def _combine_expert_outputs(self, expert_outputs: torch.Tensor,
                               gates: torch.Tensor, expert_indices: torch.Tensor,
                               num_tokens: int, d_model: int) -> torch.Tensor:
        """Combine outputs from multiple experts"""
        
        combined_output = torch.zeros(num_tokens, d_model, 
                                    device=expert_outputs.device,
                                    dtype=expert_outputs.dtype)
        
        # For each token, combine its expert outputs
        for token_idx in range(num_tokens):
            for k in range(self.top_k):
                expert_idx = expert_indices[token_idx, k]
                gate_val = gates[token_idx, k]
                
                # Add weighted expert contribution
                # Note: In practice, you'd apply the actual expert network here
                expert_output = expert_outputs[expert_idx].mean(dim=0)  # Placeholder
                combined_output[token_idx] += gate_val * expert_output
        
        return combined_output
    
    def _compute_auxiliary_losses(self, gating_logits: torch.Tensor,
                                 expert_indices: torch.Tensor,  
                                 token_counts: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute auxiliary losses for load balancing"""
        
        num_tokens = gating_logits.size(0)
        
        # Load balancing loss - encourage uniform expert usage
        expert_usage = torch.zeros(self.num_experts, device=gating_logits.device)
        for i in range(self.num_experts):
            expert_usage[i] = (expert_indices == i).float().sum()
        
        # Normalize by total tokens
        expert_usage = expert_usage / (num_tokens * self.top_k)
        
        # Compute load balance loss (encourage uniform distribution)
        uniform_usage = 1.0 / self.num_experts
        load_balance_loss = torch.sum((expert_usage - uniform_usage) ** 2)
        
        # Z-loss for stability (optional)
        gating_logits_normalized = gating_logits - gating_logits.logsumexp(dim=-1, keepdim=True)
        z_loss = torch.mean(gating_logits_normalized.logsumexp(dim=-1) ** 2)
        
        return {
            "load_balance_loss": self.load_balance_weight * load_balance_loss,
            "z_loss": 1e-4 * z_loss,
            "expert_usage": expert_usage,
            "total_tokens_per_expert": token_counts.float()
        }
    
    def get_memory_footprint(self, tokens: torch.Tensor, 
                           gating_logits: torch.Tensor) -> int:
        """Calculate memory footprint for MoE routing"""
        batch_size, seq_len, d_model = tokens.shape
        num_tokens = batch_size * seq_len
        
        # Expert capacity calculation
        expert_capacity = int(
            self.expert_capacity_factor * num_tokens * self.top_k / self.num_experts
        )
        
        # Memory for expert token storage
        expert_memory = self.num_experts * expert_capacity * d_model * 4
        
        # Memory for gates and indices
        routing_memory = num_tokens * self.top_k * 4 * 2  # gates + indices
        
        # Working memory
        working_memory = num_tokens * d_model * 4  # combined output
        
        return expert_memory + routing_memory + working_memory
    
    def get_flop_count(self, tokens: torch.Tensor,
                      gating_logits: torch.Tensor) -> int:
        """Calculate FLOP count for MoE routing"""
        batch_size, seq_len, d_model = tokens.shape
        num_tokens = batch_size * seq_len
        
        # Softmax computation
        softmax_flops = 5 * num_tokens * self.num_experts
        
        # Top-k selection (simplified estimate)
        topk_flops = num_tokens * self.num_experts * math.log2(self.num_experts)
        
        # Token dispatching and combining
        dispatch_flops = num_tokens * self.top_k * d_model
        
        return int(softmax_flops + topk_flops + dispatch_flops)
    
    def get_kernel_info(self) -> Dict[str, Any]:
        """Get MoE router kernel information"""
        info = super().get_kernel_info()
        info.update({
            "implementation": "moe_router",
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "expert_capacity_factor": self.expert_capacity_factor,
            "load_balance_weight": self.load_balance_weight
        })
        return info


class ExpertGating(nn.Module):
    """Learnable gating network for expert selection"""
    
    def __init__(self, d_model: int, num_experts: int, 
                 bias: bool = False, device: str = "cuda"):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        
        # Gating network - simple linear layer
        self.gate = nn.Linear(d_model, num_experts, bias=bias)
        
        # Initialize with small weights for stability
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)
        if bias:
            nn.init.constant_(self.gate.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute gating logits for expert selection
        
        Args:
            x: [batch_size, seq_len, d_model] input tokens
            
        Returns:
            gating_logits: [batch_size, seq_len, num_experts]
        """
        return self.gate(x)
    
    def add_noise(self, gating_logits: torch.Tensor, 
                  noise_std: float = 1.0) -> torch.Tensor:
        """Add noise to gating logits for improved load balancing"""
        if self.training:
            noise = torch.randn_like(gating_logits) * noise_std
            return gating_logits + noise
        return gating_logits 