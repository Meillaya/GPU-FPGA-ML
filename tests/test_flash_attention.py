"""Unit tests for FlashAttention implementation"""

import pytest
import torch
import numpy as np
from src.kernels.flash_attention import FlashAttention, FlashAttentionTriton


class TestFlashAttention:
    """Test suite for FlashAttention kernels"""
    
    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    @pytest.fixture
    def test_data(self, device):
        batch_size, seq_len, d_model = 2, 128, 64
        q = torch.randn(batch_size, seq_len, d_model, device=device)
        k = torch.randn(batch_size, seq_len, d_model, device=device)
        v = torch.randn(batch_size, seq_len, d_model, device=device)
        return q, k, v
    
    def test_flash_attention_init(self, device):
        """Test FlashAttention initialization"""
        flash_attn = FlashAttention(device=device, block_size=64)
        assert flash_attn.device.type == device
        assert flash_attn.block_size == 64
    
    def test_flash_attention_forward(self, device, test_data):
        """Test FlashAttention forward pass"""
        q, k, v = test_data
        flash_attn = FlashAttention(device=device, block_size=32)
        
        output = flash_attn.forward(q, k, v)
        
        assert output.shape == q.shape
        assert output.device == q.device
        assert not torch.isnan(output).any()
    
    def test_memory_footprint(self, device, test_data):
        """Test memory footprint calculation"""
        q, k, v = test_data
        flash_attn = FlashAttention(device=device, block_size=32)
        
        memory_footprint = flash_attn.get_memory_footprint(q, k, v)
        
        assert isinstance(memory_footprint, int)
        assert memory_footprint > 0
    
    def test_flop_count(self, device, test_data):
        """Test FLOP count calculation"""
        q, k, v = test_data
        flash_attn = FlashAttention(device=device, block_size=32)
        
        flop_count = flash_attn.get_flop_count(q, k, v)
        
        assert isinstance(flop_count, int)
        assert flop_count > 0
    
    def test_kernel_info(self, device):
        """Test kernel info retrieval"""
        flash_attn = FlashAttention(device=device, block_size=64)
        
        info = flash_attn.get_kernel_info()
        
        assert isinstance(info, dict)
        assert "implementation" in info
        assert "block_size" in info
        assert info["implementation"] == "flash_attention"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_benchmark(self, test_data):
        """Test benchmarking functionality"""
        q, k, v = test_data
        flash_attn = FlashAttention(device="cuda", block_size=32)
        
        metrics = flash_attn.benchmark(q, k, v, warmup_runs=2, benchmark_runs=5)
        
        assert metrics.execution_time > 0
        assert metrics.memory_used >= 0
        assert metrics.peak_memory >= 0
    
    def test_different_block_sizes(self, device, test_data):
        """Test with different block sizes"""
        q, k, v = test_data
        block_sizes = [16, 32, 64]
        
        for block_size in block_sizes:
            flash_attn = FlashAttention(device=device, block_size=block_size)
            output = flash_attn.forward(q, k, v)
            
            assert output.shape == q.shape
            assert not torch.isnan(output).any()
    
    def test_sequence_length_scaling(self, device):
        """Test scaling with different sequence lengths"""
        seq_lengths = [64, 128, 256]
        batch_size, d_model = 2, 64
        
        for seq_len in seq_lengths:
            q = torch.randn(batch_size, seq_len, d_model, device=device)
            k = torch.randn(batch_size, seq_len, d_model, device=device)  
            v = torch.randn(batch_size, seq_len, d_model, device=device)
            
            flash_attn = FlashAttention(device=device, block_size=32)
            output = flash_attn.forward(q, k, v)
            
            assert output.shape == (batch_size, seq_len, d_model)
    
    def test_output_correctness(self, device):
        """Test output correctness against standard attention"""
        batch_size, seq_len, d_model = 1, 64, 32
        
        q = torch.randn(batch_size, seq_len, d_model, device=device)
        k = torch.randn(batch_size, seq_len, d_model, device=device)
        v = torch.randn(batch_size, seq_len, d_model, device=device)
        
        # FlashAttention output
        flash_attn = FlashAttention(device=device, block_size=32, enable_fusion=False)
        flash_output = flash_attn.forward(q, k, v, causal=False)
        
        # Standard attention output
        scale = 1.0 / (d_model ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(scores, dim=-1)
        std_output = torch.matmul(attn_weights, v)
        
        # Check similarity (allowing for numerical differences)
        relative_error = torch.abs(flash_output - std_output) / (torch.abs(std_output) + 1e-8)
        assert torch.mean(relative_error) < 0.1  # 10% relative error tolerance


if __name__ == "__main__":
    pytest.main([__file__]) 