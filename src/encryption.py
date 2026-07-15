"""
CKKS homomorphic encryption utilities for encrypting model parameters
before they leave a hospital, so aggregation can happen without ever
decrypting individual local models.
"""

import tenseal as ts


def create_context():
    """Create and return a CKKS encryption context."""
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    context.global_scale = 2 ** 40
    context.generate_galois_keys()
    return context


def encrypt_model(model, context, chunk_size=4096):
    """Encrypt every parameter tensor of a model, chunked for CKKS."""
    enc_model = {}

    for k, v in model.state_dict().items():
        flat = v.flatten().tolist()

        chunks = []
        for i in range(0, len(flat), chunk_size):
            chunks.append(ts.ckks_vector(context, flat[i:i + chunk_size]))

        enc_model[k] = chunks

    return enc_model


def decrypt_model(enc, shapes):
    """Decrypt an encrypted (and aggregated) model back into tensors."""
    import torch

    dec_model = {}

    for k in enc:
        full = []
        for chunk in enc[k]:
            full.extend(chunk.decrypt())

        dec_model[k] = torch.tensor(full).view(shapes[k])

    return dec_model
