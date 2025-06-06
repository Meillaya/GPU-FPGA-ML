"""FlashAttention Implementation

Memory-efficient attention mechanism that reduces memory complexity
from O(N²) to O(N) through tiling and recomputation.

References:
- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
- FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Tuple
import math

from .base_kernel import BaseKernel, KernelMetrics


@triton.jit
def flash_attention_kernel(
    Q, K, V, O,  # Input/Output tensors
    L, M,  # Logsumexp and max values for numerical stability
    seq_len, d_head, 
    block_size_q: tl.constexpr, block_size_k: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """Triton implementation of FlashAttention kernel"""
    
    # Get program IDs
    start_q = tl.program_id(0) * block_size_q
    start_k = tl.program_id(1) * block_size_k
    
    # Create offset ranges for Q, K, V blocks
    offs_q = start_q + tl.arange(0, block_size_q)
    offs_k = start_k + tl.arange(0, block_size_k)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Load Q block
    q_ptrs = Q + offs_q[:, None] * d_head + offs_d[None, :]
    q = tl.load(q_ptrs, mask=(offs_q[:, None] < seq_len))
    
    # Initialize output accumulator and normalization stats
    o = tl.zeros([block_size_q, BLOCK_DMODEL], dtype=tl.float32)
    l = tl.zeros([block_size_q], dtype=tl.float32)
    m = tl.full([block_size_q], -float('inf'), dtype=tl.float32)
    
    # Loop over K, V blocks
    for start_k in range(0, seq_len, block_size_k):
        offs_k = start_k + tl.arange(0, block_size_k)
        
        # Load K, V blocks
        k_ptrs = K + offs_k[None, :] * d_head + offs_d[:, None]
        v_ptrs = V + offs_k[:, None] * d_head + offs_d[None, :]
        
        k = tl.load(k_ptrs, mask=(offs_k[None, :] < seq_len))
        v = tl.load(v_ptrs, mask=(offs_k[:, None] < seq_len))
        
        # Compute attention scores
        qk = tl.dot(q, k)  # [block_size_q, block_size_k]
        qk = qk * (1.0 / math.sqrt(d_head))
        
        # Apply causal mask (for decoder attention)
        mask = offs_q[:, None] >= offs_k[None, :]
        qk = tl.where(mask, qk, -float('inf'))
        
        # Online softmax with numerical stability
        m_new = tl.maximum(m, tl.max(qk, axis=1))
        alpha = tl.exp(m - m_new)
        beta = tl.exp(qk - m_new[:, None])
        
        # Update output and normalization
        l_new = alpha * l + tl.sum(beta, axis=1)
        o = o * alpha[:, None] + tl.dot(beta, v)
        
        # Update stats
        l = l_new
        m = m_new
    
    # Final normalization
    o = o / l[:, None]
    
    # Store output
    o_ptrs = O + offs_q[:, None] * d_head + offs_d[None, :]
    tl.store(o_ptrs, o, mask=(offs_q[:, None] < seq_len))
    
    # Store logsumexp for backward pass
    l_ptrs = L + offs_q
    m_ptrs = M + offs_q
    tl.store(l_ptrs, l, mask=(offs_q < seq_len))
    tl.store(m_ptrs, m, mask=(offs_q < seq_len))


class FlashAttentionTriton(BaseKernel):
    """Triton-based FlashAttention implementation"""
    
    def __init__(self, device: str = "cuda", block_size: int = 64):
        super().__init__(device)
        self.block_size = block_size
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                causal: bool = True) -> torch.Tensor:
        """
        Forward pass of FlashAttention
        
        Args:
            q, k, v: [batch_size, seq_len, d_head] attention tensors
            causal: Whether to apply causal masking
            
        Returns:
            output: [batch_size, seq_len, d_head] attention output
        """
        batch_size, seq_len, d_head = q.shape
        
        # Allocate output and intermediate tensors
        output = torch.empty_like(q)
        L = torch.empty((batch_size, seq_len), device=q.device, dtype=torch.float32)
        M = torch.empty((batch_size, seq_len), device=q.device, dtype=torch.float32)
        
        # Launch kernel
        grid = (triton.cdiv(seq_len, self.block_size), 1)
        
        flash_attention_kernel[grid](
            q, k, v, output, L, M,
            seq_len, d_head,
            block_size_q=self.block_size,
            block_size_k=self.block_size,
            BLOCK_DMODEL=d_head,
        )
        
        return output
    
    def get_memory_footprint(self, q: torch.Tensor, k: torch.Tensor, 
                           v: torch.Tensor) -> int:
        """Calculate memory footprint"""
        batch_size, seq_len, d_head = q.shape
        
        # Input tensors
        input_memory = 3 * batch_size * seq_len * d_head * 4  # float32
        
        # Output tensor  
        output_memory = batch_size * seq_len * d_head * 4
        
        # Intermediate tensors (L, M)
        intermediate_memory = 2 * batch_size * seq_len * 4
        
        # Working memory (blocks)
        block_memory = 3 * self.block_size * d_head * 4
        
        return input_memory + output_memory + intermediate_memory + block_memory
    
    def get_flop_count(self, q: torch.Tensor, k: torch.Tensor, 
                      v: torch.Tensor) -> int:
        """Calculate FLOP count"""
        batch_size, seq_len, d_head = q.shape
        
        # QK^T computation: batch_size * seq_len^2 * d_head
        qk_flops = batch_size * seq_len * seq_len * d_head
        
        # Softmax: approximately 5 * batch_size * seq_len^2  
        softmax_flops = 5 * batch_size * seq_len * seq_len
        
        # Attention * V: batch_size * seq_len^2 * d_head
        av_flops = batch_size * seq_len * seq_len * d_head
        
        return qk_flops + softmax_flops + av_flops


class FlashAttention(BaseKernel):
    """CUDA-based FlashAttention with kernel fusion optimizations"""
    
    def __init__(self, device: str = "cuda", block_size: int = 64,
                 enable_fusion: bool = True):
        super().__init__(device)
        self.block_size = block_size
        self.enable_fusion = enable_fusion
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                causal: bool = True) -> torch.Tensor:
        """
        Memory-efficient attention forward pass
        
        Implementation uses tiling to reduce memory from O(N²) to O(N)
        """
        batch_size, seq_len, d_head = q.shape
        scale = 1.0 / math.sqrt(d_head)
        
        if self.enable_fusion and hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch's fused implementation when available
            return F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=attn_mask,
                is_causal=causal,
                scale=scale
            )
        else:
            # Fallback to manual tiled implementation
            return self._tiled_attention(q, k, v, scale, causal, attn_mask)
    
    def _tiled_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        scale: float, causal: bool, 
                        attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Manual tiled attention implementation for educational purposes"""
        batch_size, seq_len, d_head = q.shape
        
        # Initialize output and running statistics
        output = torch.zeros_like(q)
        row_max = torch.full((batch_size, seq_len), -float('inf'), 
                           device=q.device, dtype=torch.float32)
        row_sum = torch.zeros((batch_size, seq_len), 
                            device=q.device, dtype=torch.float32)
        
        # Process in blocks to reduce memory usage
        block_size = min(self.block_size, seq_len)
        
        for i in range(0, seq_len, block_size):
            q_block = q[:, i:i+block_size]  # [B, block_size, D]
            
            # Initialize block output and statistics
            block_output = torch.zeros_like(q_block)
            block_max = torch.full((batch_size, q_block.size(1)), -float('inf'),
                                 device=q.device, dtype=torch.float32)
            block_sum = torch.zeros((batch_size, q_block.size(1)),
                                  device=q.device, dtype=torch.float32)
            
            for j in range(0, seq_len, block_size):
                k_block = k[:, j:j+block_size]  # [B, block_size, D]
                v_block = v[:, j:j+block_size]  # [B, block_size, D]
                
                # Compute attention scores
                scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale
                
                # Apply causal mask
                if causal:
                    mask = torch.triu(torch.ones(q_block.size(1), k_block.size(1),
                                               device=q.device), diagonal=j-i+1)
                    scores.masked_fill_(mask.bool(), -float('inf'))
                
                # Apply attention mask if provided
                if attn_mask is not None:
                    scores += attn_mask[:, i:i+q_block.size(1), j:j+k_block.size(1)]
                
                # Online softmax computation
                block_max_new = torch.maximum(block_max, scores.max(dim=-1).values)
                scores_exp = torch.exp(scores - block_max_new.unsqueeze(-1))
                
                # Update running sums
                scale_factor = torch.exp(block_max - block_max_new)
                block_output = block_output * scale_factor.unsqueeze(-1)
                block_sum = block_sum * scale_factor
                
                # Add current contribution
                block_output += torch.matmul(scores_exp, v_block)
                block_sum += scores_exp.sum(dim=-1)
                block_max = block_max_new
            
            # Normalize and store
            output[:, i:i+block_size] = block_output / block_sum.unsqueeze(-1)
        
        return output
    
    def get_memory_footprint(self, q: torch.Tensor, k: torch.Tensor,
                           v: torch.Tensor) -> int:
        """Calculate memory footprint for FlashAttention"""
        batch_size, seq_len, d_head = q.shape
        
        # Standard attention would need O(N²) for attention matrix
        standard_memory = batch_size * seq_len * seq_len * 4
        
        # FlashAttention needs O(N) memory
        flash_memory = batch_size * seq_len * d_head * 4  # Output
        flash_memory += 2 * batch_size * seq_len * 4  # Running stats
        flash_memory += self.block_size * self.block_size * 4  # Block buffer
        
        return flash_memory
    
    def get_flop_count(self, q: torch.Tensor, k: torch.Tensor,
                      v: torch.Tensor) -> int:
        """Calculate theoretical FLOP count"""
        batch_size, seq_len, d_head = q.shape
        return 2 * batch_size * seq_len * seq_len * d_head  # Same as standard attention
    
    def get_kernel_info(self) -> dict:
        """Get FlashAttention kernel information"""
        info = super().get_kernel_info()
        info.update({
            "implementation": "flash_attention",
            "block_size": self.block_size,
            "fusion_enabled": self.enable_fusion,
            "memory_complexity": "O(N)",
            "compute_complexity": "O(N²)"
        })
        return info 