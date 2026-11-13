#!/usr/bin/env python3
"""Validate actual key serialization, encryption, and partial decryption."""
import argparse
import json
import math
import torch
import norm_benchmark as bench


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=['wo_Dist','wo_Lazy','wo_Coeff','wo_Ext','ARMOR'],required=True)
    parser.add_argument('--ringdim',type=int,choices=[16384,32768,65536],default=16384)
    args=parser.parse_args()
    method=args.method
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.cuda.set_device(0);torch.set_num_threads(1)
    spec=bench.specification(method,args.ringdim)
    context=bench.make_context(spec)
    master=bench.phantom.secret_key(context)
    s1,s2,q1,q2=bench.generate_key_shares(master,context,method)
    material=dict(public_key=bench.phantom.public_key.from_bytes(master.gen_publickey(context).to_bytes()),
                  encoder=bench.phantom.ckks_encoder(context),
                  secret_share=bench.phantom.secret_key.from_bytes(s1.to_bytes()),
                  product_share=bench.phantom.secret_key.from_bytes(q1.to_bytes()))
    remote=dict(secret_share=bench.phantom.secret_key.from_bytes(s2.to_bytes()),
                product_share=bench.phantom.secret_key.from_bytes(q2.to_bytes()))
    if method in bench.KEYSWITCH:
        material['relin_key']=bench.phantom.relin_key.from_bytes(master.gen_relinkey(context).to_bytes())
        material['galois_key']=bench.phantom.galois_key.from_bytes(master.create_galois_keys(context).to_bytes())
    generator=torch.Generator().manual_seed(2023)
    vector=torch.randn(args.ringdim+3,dtype=torch.float64,generator=generator)*.01
    original,_=bench.encrypt_vector(method,context,spec,material,vector)
    original=bench.unpack_ciphertexts(bench.pack_ciphertexts(original))
    groups=[[ct] for ct in original] if method in bench.PER_CHUNK else [original]
    total=0.
    for chunks in groups:
        product=bench.compute_job(method,context,spec,material,chunks,[])
        product=bench.phantom.ciphertext.from_bytes(product.to_bytes())
        local=bench.partial_norm(method,context,spec,material,product,False)
        partial=bench.partial_norm(method,context,spec,remote,product,True)
        if method in bench.FULL_PARTIAL:partial=partial.to_bytes()
        total+=bench.decode_norm(method,context,spec,material,local,partial)
    expected=float(vector.square().sum())
    if not math.isclose(total,expected,rel_tol=1e-3,abs_tol=1e-3):
        raise AssertionError(f'Norm mismatch: actual={total}, expected={expected}')
    print(json.dumps(dict(method=method,status='passed',expected=expected,actual=total,absolute_error=abs(total-expected))))


if __name__=='__main__':main()
