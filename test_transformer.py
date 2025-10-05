"""
Unit Tests untuk Implementasi Transformer
==========================================

Testing komprehensif untuk setiap komponen Transformer
Author: Adinda Putri Romadhon
NIM: 22/505508/TK/55321
Mata Kuliah: Pemrosesan Bahasa Alami
"""

import numpy as np
import sys

# Import semua komponen dari transformer.py
from transformer import (
    TokenEmbedding,
    PositionalEncoding,
    ScaledDotProductAttention,
    MultiHeadAttention,
    FeedForwardNetwork,
    LayerNormalization,
    DecoderLayer,
    DecoderOnlyTransformer
)


def test_token_embedding():
    """Test 1: Token Embedding Layer"""
    print("\n" + "="*70)
    print("TEST 1: Token Embedding")
    print("="*70)
    
    vocab_size = 100
    d_model = 64
    batch_size = 2
    seq_len = 10
    
    # Inisialisasi
    embedding = TokenEmbedding(vocab_size, d_model, seed=42)
    token_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
    
    # Forward pass
    output = embedding.forward(token_ids)
    
    # Validasi shape
    expected_shape = (batch_size, seq_len, d_model)
    assert output.shape == expected_shape, f"Shape error! Expected {expected_shape}, got {output.shape}"
    
    # Validasi scaling
    print(f"✓ Input shape: {token_ids.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Scaling factor: √{d_model} = {np.sqrt(d_model):.2f}")
    print(f"✓ Output statistics:")
    print(f"    Mean: {output.mean():.4f}")
    print(f"    Std: {output.std():.4f}")
    print(f"    Range: [{output.min():.4f}, {output.max():.4f}]")
    print("Token Embedding: PASSED")
    
    return True


def test_positional_encoding():
    """Test 2: Positional Encoding"""
    print("\n" + "="*70)
    print("TEST 2: Positional Encoding")
    print("="*70)
    
    d_model = 64
    max_seq_len = 512
    batch_size = 2
    seq_len = 10
    
    # Inisialisasi
    pos_enc = PositionalEncoding(d_model, max_seq_len)
    
    # Cek precomputed PE
    assert pos_enc.pe.shape == (max_seq_len, d_model), "PE shape incorrect"
    
    # Test forward pass
    x = np.random.randn(batch_size, seq_len, d_model)
    output = pos_enc.forward(x)
    
    assert output.shape == x.shape, "Output shape mismatch"
    
    # Verifikasi sinusoidal pattern
    pe_sample = pos_enc.pe[:5, :4]
    print(f"✓ PE shape: {pos_enc.pe.shape}")
    print(f"✓ Sample PE (first 5 pos, 4 dims):")
    print(pe_sample)
    
    # Cek even/odd indices (sin/cos)
    even_idx = pos_enc.pe[:, 0::2]
    odd_idx = pos_enc.pe[:, 1::2]
    print(f"✓ Even indices (sin) range: [{even_idx.min():.3f}, {even_idx.max():.3f}]")
    print(f"✓ Odd indices (cos) range: [{odd_idx.min():.3f}, {odd_idx.max():.3f}]")
    
    # Validasi range [-1, 1]
    assert even_idx.min() >= -1.1 and even_idx.max() <= 1.1, "Sin values out of range"
    assert odd_idx.min() >= -1.1 and odd_idx.max() <= 1.1, "Cos values out of range"
    
    print("Positional Encoding: PASSED")
    
    return True


def test_scaled_dot_product_attention():
    """Test 3: Scaled Dot-Product Attention"""
    print("\n" + "="*70)
    print("TEST 3: Scaled Dot-Product Attention")
    print("="*70)
    
    batch_size = 2
    seq_len = 5
    d_k = 8
    
    attention = ScaledDotProductAttention()
    
    # Random Q, K, V
    Q = np.random.randn(batch_size, seq_len, d_k)
    K = np.random.randn(batch_size, seq_len, d_k)
    V = np.random.randn(batch_size, seq_len, d_k)
    
    # Test without mask
    output = attention.forward(Q, K, V)
    assert output.shape == (batch_size, seq_len, d_k), "Output shape incorrect"
    
    # Test with causal mask
    mask = np.tril(np.ones((seq_len, seq_len)))
    output_masked = attention.forward(Q, K, V, mask)
    
    # Cek attention weights
    attn_weights = attention.attention_weights
    print(f"✓ Q, K, V shapes: {Q.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Attention weights shape: {attn_weights.shape}")
    
    # Verifikasi sum = 1
    attn_sum = np.sum(attn_weights, axis=-1)
    assert np.allclose(attn_sum, 1.0), "Attention weights don't sum to 1"
    print(f"✓ Attention weights sum: {attn_sum[0, 0]:.6f} (should be 1.0)")
    
    # Verifikasi causal mask
    upper_triangle = attn_weights[0, 0, :] * (1 - mask[0, :])
    upper_sum = np.sum(upper_triangle)
    print(f"✓ Upper triangle sum: {upper_sum:.8f} (should be ~0)")
    assert upper_sum < 1e-6, "Causal mask not working properly"
    
    print("Scaled Dot-Product Attention: PASSED")
    
    return True


def test_multi_head_attention():
    """Test 4: Multi-Head Attention"""
    print("\n" + "="*70)
    print("TEST 4: Multi-Head Attention")
    print("="*70)
    
    d_model = 64
    num_heads = 4
    batch_size = 2
    seq_len = 10
    
    mha = MultiHeadAttention(d_model, num_heads, seed=42)
    
    x = np.random.randn(batch_size, seq_len, d_model)
    mask = np.tril(np.ones((seq_len, seq_len)))
    
    output = mha.forward(x, mask)
    
    assert output.shape == x.shape, "Output shape mismatch"
    assert mha.d_k == d_model // num_heads, "d_k calculation wrong"
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Number of heads: {num_heads}")
    print(f"✓ d_k per head: {mha.d_k}")
    print(f"✓ Weight matrices:")
    print(f"    W_q: {mha.W_q.shape}")
    print(f"    W_k: {mha.W_k.shape}")
    print(f"    W_v: {mha.W_v.shape}")
    print(f"    W_o: {mha.W_o.shape}")
    
    print("Multi-Head Attention: PASSED")
    
    return True


def test_feed_forward_network():
    """Test 5: Feed-Forward Network"""
    print("\n" + "="*70)
    print("TEST 5: Feed-Forward Network")
    print("="*70)
    
    d_model = 64
    d_ff = 256
    batch_size = 2
    seq_len = 10
    
    ffn = FeedForwardNetwork(d_model, d_ff, seed=42)
    
    x = np.random.randn(batch_size, seq_len, d_model)
    output = ffn.forward(x)
    
    assert output.shape == x.shape, "Output shape mismatch"
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Hidden dimension: {d_ff}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Weight shapes:")
    print(f"    W1: {ffn.W1.shape} (d_model → d_ff)")
    print(f"    W2: {ffn.W2.shape} (d_ff → d_model)")
    print(f"✓ Bias shapes:")
    print(f"    b1: {ffn.b1.shape}")
    print(f"    b2: {ffn.b2.shape}")
    
    # Test ReLU activation
    hidden = np.matmul(x, ffn.W1) + ffn.b1
    relu_hidden = np.maximum(0, hidden)
    num_zeros = np.sum(relu_hidden == 0)
    print(f"✓ ReLU activated: {num_zeros} zeros out of {relu_hidden.size} elements")
    
    print("Feed-Forward Network: PASSED")
    
    return True


def test_layer_normalization():
    """Test 6: Layer Normalization"""
    print("\n" + "="*70)
    print("TEST 6: Layer Normalization")
    print("="*70)
    
    d_model = 64
    batch_size = 2
    seq_len = 10
    
    ln = LayerNormalization(d_model)
    
    # Input dengan scale besar
    x = np.random.randn(batch_size, seq_len, d_model) * 10 + 5
    output = ln.forward(x)
    
    # Cek normalization
    mean = np.mean(output, axis=-1)
    std = np.std(output, axis=-1)
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Before normalization:")
    print(f"    Mean: {x.mean():.4f}, Std: {x.std():.4f}")
    print(f"✓ After normalization:")
    print(f"    Mean: {output.mean():.6f}, Std: {output.std():.4f}")
    print(f"✓ Per-sample statistics:")
    print(f"    Mean range: [{mean.min():.6f}, {mean.max():.6f}]")
    print(f"    Std range: [{std.min():.4f}, {std.max():.4f}]")
    
    # Validasi
    assert np.allclose(mean, 0, atol=1e-5), "Mean not close to 0"
    assert np.allclose(std, 1, atol=0.2), "Std not close to 1"
    
    print("Layer Normalization: PASSED")
    
    return True


def test_decoder_layer():
    """Test 7: Decoder Layer"""
    print("\n" + "="*70)
    print("TEST 7: Decoder Layer")
    print("="*70)
    
    d_model = 64
    num_heads = 4
    d_ff = 256
    batch_size = 2
    seq_len = 10
    
    decoder = DecoderLayer(d_model, num_heads, d_ff, seed=42)
    
    x = np.random.randn(batch_size, seq_len, d_model)
    mask = np.tril(np.ones((seq_len, seq_len)))
    
    output = decoder.forward(x, mask)
    
    assert output.shape == x.shape, "Output shape mismatch"
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Components:")
    print(f"    Multi-Head Attention: {type(decoder.mha).__name__}")
    print(f"    Feed-Forward Network: {type(decoder.ffn).__name__}")
    print(f"    Layer Norm 1: {type(decoder.ln1).__name__}")
    print(f"    Layer Norm 2: {type(decoder.ln2).__name__}")
    
    print("Decoder Layer: PASSED")
    
    return True


def test_causal_mask():
    """Test 8: Causal Mask"""
    print("\n" + "="*70)
    print("TEST 8: Causal Mask")
    print("="*70)
    
    model = DecoderOnlyTransformer(
        vocab_size=100,
        d_model=64,
        num_heads=4,
        num_layers=2,
        seed=42
    )
    
    # Test berbagai ukuran
    for seq_len in [5, 10, 15]:
        mask = model.create_causal_mask(seq_len)
        
        # Verifikasi lower triangular
        is_lower_tri = np.allclose(mask, np.tril(mask))
        assert is_lower_tri, f"Mask for seq_len={seq_len} is not lower triangular"
        
        # Verifikasi diagonal
        diagonal_sum = np.trace(mask)
        assert diagonal_sum == seq_len, f"Diagonal sum should be {seq_len}"
        
        print(f"✓ seq_len={seq_len}: Lower triangular ✓, Diagonal sum={int(diagonal_sum)} ✓")
    
    # Tampilkan contoh mask
    print(f"\n✓ Sample mask (6x6):")
    sample_mask = model.create_causal_mask(6)
    print(sample_mask.astype(int))
    
    print("Causal Mask: PASSED")
    
    return True


def test_full_transformer():
    """Test 9: Full Transformer Forward Pass"""
    print("\n" + "="*70)
    print("TEST 9: Full Transformer")
    print("="*70)
    
    vocab_size = 100
    d_model = 64
    num_heads = 4
    num_layers = 2
    d_ff = 256
    batch_size = 2
    seq_len = 10
    
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        seed=42
    )
    
    token_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
    logits, probs = model.forward(token_ids)
    
    # Validasi shapes
    assert logits.shape == (batch_size, seq_len, vocab_size), "Logits shape incorrect"
    assert probs.shape == (batch_size, vocab_size), "Probs shape incorrect"
    
    # Validasi probabilitas
    prob_sum = np.sum(probs, axis=-1)
    assert np.allclose(prob_sum, 1.0), "Probabilities don't sum to 1"
    
    # Validasi no NaN/Inf
    assert not np.isnan(logits).any(), "NaN in logits"
    assert not np.isinf(logits).any(), "Inf in logits"
    assert not np.isnan(probs).any(), "NaN in probs"
    assert not np.isinf(probs).any(), "Inf in probs"
    
    print(f"✓ Model configuration:")
    print(f"    Vocab size: {vocab_size}")
    print(f"    d_model: {d_model}")
    print(f"    num_heads: {num_heads}")
    print(f"    num_layers: {num_layers}")
    print(f"    d_ff: {d_ff}")
    print(f"✓ Input shape: {token_ids.shape}")
    print(f"✓ Logits shape: {logits.shape}")
    print(f"✓ Probs shape: {probs.shape}")
    print(f"✓ Probability sum: {prob_sum[0]:.6f} (should be 1.0)")
    print(f"✓ Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
    print(f"✓ Probs range: [{probs.min():.6f}, {probs.max():.6f}]")
    
    # Top-5 predictions
    top_5_idx = np.argsort(probs[0])[-5:][::-1]
    top_5_probs = probs[0][top_5_idx]
    print(f"✓ Top-5 predictions:")
    for i, (idx, prob) in enumerate(zip(top_5_idx, top_5_probs), 1):
        print(f"    {i}. Token {idx}: {prob:.6f} ({prob*100:.2f}%)")
    
    print("Full Transformer: PASSED")
    
    return True


def test_different_configurations():
    """Test 10: Different Batch Sizes and Sequence Lengths"""
    print("\n" + "="*70)
    print("TEST 10: Different Configurations")
    print("="*70)
    
    vocab_size = 50
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=32,
        num_heads=2,
        num_layers=1,
        d_ff=128,
        seed=42
    )
    
    test_configs = [
        (1, 5),   # batch=1, seq=5
        (2, 10),  # batch=2, seq=10
        (4, 8),   # batch=4, seq=8
        (1, 20),  # batch=1, seq=20
    ]
    
    for batch_size, seq_len in test_configs:
        token_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
        logits, probs = model.forward(token_ids)
        
        assert logits.shape == (batch_size, seq_len, vocab_size)
        assert probs.shape == (batch_size, vocab_size)
        assert np.allclose(np.sum(probs, axis=-1), 1.0)
        
        print(f"✓ Batch={batch_size}, Seq={seq_len}: {logits.shape} → {probs.shape} ✓")
    
    print("Different Configurations: ALL PASSED")
    
    return True


def run_all_tests():
    """Jalankan semua tests"""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  COMPREHENSIVE UNIT TESTS - TRANSFORMER  ".center(68) + "█")
    print("█" + "  Author: Adinda Putri Romadhon  ".center(68) + "█")
    print("█" + "  NIM: 22/505508/TK/55321  ".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    tests = [
        ("Token Embedding", test_token_embedding),
        ("Positional Encoding", test_positional_encoding),
        ("Scaled Dot-Product Attention", test_scaled_dot_product_attention),
        ("Multi-Head Attention", test_multi_head_attention),
        ("Feed-Forward Network", test_feed_forward_network),
        ("Layer Normalization", test_layer_normalization),
        ("Decoder Layer", test_decoder_layer),
        ("Causal Mask", test_causal_mask),
        ("Full Transformer", test_full_transformer),
        ("Different Configurations", test_different_configurations),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ {test_name} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ {test_name} ERROR: {e}")
            failed += 1
    
    # Summary
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    if failed == 0:
        print("█" + "  🎉 ALL TESTS PASSED! 🎉  ".center(68) + "█")
    else:
        print("█" + f"  ⚠️  {passed} PASSED, {failed} FAILED  ".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    if failed == 0:
        print("\nSUKSES! Implementasi sudah benar")
        print("SUKSES! Semua komponen bekerja dengan baik!")
        print("SUKSES! Dimensi tensor sudah correct!")
        print("SUKSES! Causal mask berfungsi!")
        print("SUKSES! Probabilitas valid (sum = 1)!")
        print("\nTotal Tests: {}/{}".format(passed, len(tests)))
    else:
        print(f"\nMasih ada {failed} test yang gagal. Silakan perbaiki.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)