"""Four-role norm-only comparison. Algorithms follow local wo_Dist/wo_Lazy/wo_Ext/Ours/slot references.

No FL training, layer sampling or dummy network-size estimates. All frames use
actual serialized ciphertexts/key shares and authenticated multiprocessing TCP.
"""
import argparse
import ctypes
import json
import hashlib
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = next(parent for parent in ROOT.parents if (parent / 'Ciphertext').is_dir())
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'LAN_WAN'))
NATIVE = PROJECT_ROOT / 'phantom-fhe' / 'build' / 'lib'
if (NATIVE / 'libPhantom.so').exists():
    ctypes.CDLL(str(NATIVE / 'libPhantom.so'), mode=ctypes.RTLD_GLOBAL)
    sys.path.insert(0, str(NATIVE))
import pyPhantom as phantom
import numpy as np
import torch
from Ciphertext.encrypt_decrypt import encrypt
from distributed import (MeteredConnection, connect, listen, make_context,
                         pack_ciphertexts, unpack_ciphertexts, log_event)

METHODS = ('wo_Dist', 'wo_Lazy', 'wo_Ext', 'wo_Coeff', 'Yasuda', 'ARMOR')
ROLES = ('keycenter', 'server1', 'server2', 'client')
KEYSWITCH = {'wo_Dist', 'wo_Coeff'}
PER_CHUNK = {'wo_Lazy'}
FULL_PARTIAL = {'wo_Ext', 'Yasuda', 'wo_Coeff'}
INPUT_BLOCK_SIZE = 16384
INPUT_BLOCKS = (62, 184, 367)


def parser(method):
    p = argparse.ArgumentParser(description=f'{method}: norm-only four-process benchmark')
    p.add_argument('--input_blocks', type=int, nargs='+', choices=INPUT_BLOCKS, default=list(INPUT_BLOCKS),
                   help='Random input lengths in units of 16384; default sweeps all three sizes')
    p.add_argument('--ringdim', type=int, default=16384)
    p.add_argument('--iterations', type=int, default=5, help='Measured iterations after warmup')
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--num_clients', type=int, default=1, help='Logical clients executed sequentially')
    p.add_argument('--seed', type=int, default=2023)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--output', default='runs/norm_benchmarks')
    p.add_argument('--run_tag', default='rep1', help='Use a fresh tag per run; logs append')
    p.add_argument('--rtol', type=float, default=1e-3)
    p.add_argument('--atol', type=float, default=1e-3)
    p.add_argument('--keycenter_host', default='127.0.0.1')
    p.add_argument('--keycenter_port', type=int, default=9100)
    p.add_argument('--server1_host', default='127.0.0.1')
    p.add_argument('--server1_port', type=int, default=9101)
    p.add_argument('--server2_host', default='127.0.0.1')
    p.add_argument('--server2_port', type=int, default=9102)
    p.add_argument('--connect_timeout', type=float, default=600)
    return p


def validate(args):
    if (args.ringdim < 1024 or args.ringdim & (args.ringdim-1)
            or args.iterations < 1 or args.warmup < 0 or args.num_clients < 1 or args.threads < 1):
        raise ValueError('Positive sizes/iterations required; ringdim must be a power of two >= 1024')
    if len(args.input_blocks) != len(set(args.input_blocks)):
        raise ValueError('Do not repeat input sizes in one run')
    if args.gpu < 0 or not torch.cuda.is_available():
        raise RuntimeError('Every norm benchmark role requires a working CUDA GPU')
    if not all(math.isfinite(v) and v >= 0 for v in (args.rtol,args.atol)):
        raise ValueError('Finite nonnegative tolerances required')
    torch.cuda.set_device(args.gpu)
    torch.set_num_threads(args.threads)


def specification(method, degree):
    switched = method in KEYSWITCH
    steps = [0]
    if method == 'wo_Coeff':
        steps += list(range(1,16)) + [2**i for i in range(4,int(math.log2(degree//2)))]
    return dict(ringdim=degree, modulus_chain=[62,52,52,52]+([60] if switched else []),
                size_P=int(switched), galois_steps=steps, scale=2.**88,
                is_version2=False, packing='slots' if method=='wo_Coeff' else 'coefficients')


def configure_log(args, method, role):
    directory=Path(args.output)/method/args.run_tag
    if not directory.is_absolute():
        directory=ROOT/directory
    directory.mkdir(parents=True,exist_ok=True)
    args.log_path=str(directory/f'{role}.jsonl')
    return directory


def open_peer(host,port,timeout):
    conn=connect(host,port,timeout)
    if conn is None:
        raise TimeoutError(f'Cannot connect to {host}:{port} within {timeout}s')
    return conn


def expect(conn,kind,round_idx=None):
    message=conn.recv()
    if message.get('type')!=kind or (round_idx is not None and message.get('round')!=round_idx):
        raise ValueError(f'Expected {kind}, round={round_idx}; received {message.get("type")}, {message.get("round")}')
    return message


def measured(function,*args):
    phantom.cudaStreamSychronize()
    start=time.perf_counter()
    value=function(*args)
    phantom.cudaStreamSychronize()
    return value,time.perf_counter()-start


def key_packet(method,role,config,spec,public,shares,evaluation):
    if role not in {'client','server1','server2'}:
        raise ValueError('Unknown role')
    result=dict(type='benchmark_keys',method=method,args=config,spec=spec,public_key=public)
    if role in {'server1','server2'}:
        result['secret_share']=shares[role][0]
        if method in {'ARMOR','wo_Lazy','wo_Ext','Yasuda'}:
            result['product_share']=shares[role][1]
    if role=='server1':
        result.update(evaluation)
    return result


def generate_key_shares(master, context, method):
    """Respect the native binding signature without substituting key protocols."""
    signature = getattr(type(master).armor_gen_keys, '__doc__', '') or ''
    if 'bool' in signature:
        return master.armor_gen_keys(context, method == 'Yasuda')
    if method == 'Yasuda':
        raise RuntimeError('Yasuda requires native key generation with square-key shares; this build exposes ARMOR shares only')
    return master.armor_gen_keys(context)


def run_keycenter(method,args):
    spec=specification(method,args.ringdim)
    start=time.perf_counter()
    context=make_context(spec)
    master=phantom.secret_key(context)
    s1,s2,q1,q2=generate_key_shares(master,context,method)
    public=master.gen_publickey(context).to_bytes()
    shares={'server1':(s1.to_bytes(),q1.to_bytes()),'server2':(s2.to_bytes(),q2.to_bytes())}
    evaluation={}
    if method in KEYSWITCH:
        evaluation=dict(relin_key=master.gen_relinkey(context).to_bytes(),
                        galois_key=master.create_galois_keys(context).to_bytes())
    phantom.cudaStreamSychronize()
    log_event(args.log_path,'keycenter','key_generation',method=method,latency_s=time.perf_counter()-start,spec=spec)
    config={k:v for k,v in vars(args).items() if k!='log_path'}
    waiting={'server1','server2','client'}
    with listen(args.keycenter_host,args.keycenter_port) as listener:
        print(f'[{method}/KeyCenter] listening {listener.address}',flush=True)
        while waiting:
            with MeteredConnection(listener.accept(),args.log_path,'keycenter','participant') as conn:
                hello=expect(conn,'hello')
                role=hello.get('role')
                if role not in waiting or hello.get('method')!=method:
                    conn.send(dict(type='error',message='Method mismatch or duplicate/unknown role'))
                    continue
                conn.peer=role
                conn.send(key_packet(method,role,config,spec,public,shares,evaluation))
                waiting.remove(role)
    print(f'[{method}/KeyCenter] key distribution complete',flush=True)


def get_material(method,local,role):
    with MeteredConnection(open_peer(local.keycenter_host,local.keycenter_port,local.connect_timeout),
                           local.log_path,role,'keycenter') as conn:
        conn.send(dict(type='hello',role=role,method=method))
        packet=expect(conn,'benchmark_keys')
    if packet['method']!=method:
        raise ValueError('Methods differ between roles')
    args=SimpleNamespace(**packet['args'])
    args.gpu=local.gpu
    validate(args)
    configure_log(args,method,role)
    spec=packet['spec']
    context=make_context(spec)
    material=dict(encoder=phantom.ckks_encoder(context),
                  public_key=phantom.public_key.from_bytes(packet['public_key']))
    for name in ('secret_share','product_share'):
        if name in packet:
            material[name]=phantom.secret_key.from_bytes(packet[name])
    if 'relin_key' in packet:
        material['relin_key']=phantom.relin_key.from_bytes(packet['relin_key'])
        material['galois_key']=phantom.galois_key.from_bytes(packet['galois_key'])
    return args,context,spec,material


def input_vector(args, blocks):
    """Same seeded random vector for every method; no model/dataset is loaded."""
    if blocks not in INPUT_BLOCKS:
        raise ValueError('Expected 62, 184 or 367 input blocks')
    generator=np.random.default_rng(args.seed)
    return torch.from_numpy(generator.normal(0., .01, blocks*INPUT_BLOCK_SIZE))


def input_digest(vector):
    return hashlib.sha256(vector.numpy().tobytes()).hexdigest()


def encrypt_vector(method,context,spec,material,vector):
    degree=spec['ringdim']//2 if method=='wo_Coeff' else spec['ringdim']
    chunks=encrypt.dividing_by_chunk(vector,SimpleNamespace(ringdim=degree))
    original=[]
    encoder,public=material['encoder'],material['public_key']
    for chunk in chunks:
        if method=='wo_Coeff':
            plain=encoder.encode_double_vector(context,chunk.numpy().tolist(),spec['scale'],spec['size_P'])
        else:
            plain=encoder.armor_coefficient_encode(context,chunk.numpy(),spec['scale'],spec['size_P'])
        original.append(public.armor_encrypt_asymmetric(context,plain))
    inverse=[]
    if method=='Yasuda':
        for coefficients in encrypt.conjugate_rotate_form_in_list(chunks):
            plain=encoder.armor_coefficient_encode(context,coefficients,spec['scale'],0)
            inverse.append(public.armor_encrypt_asymmetric(context,plain))
    return original,inverse


def finish_product(context,value,spec):
    value=phantom.armor_to_intt(context,value)
    # Special primes are already absent at the encoding/data level.
    # armor_rescale_to takes a drop count, and the decoder needs two data limbs.
    value=phantom.armor_rescale_to(context,value,len(spec['modulus_chain'])-spec['size_P']-2)
    return phantom.armor_to_ntt(context,value)


def compute_job(method,context,spec,material,original,inverse):
    """One job for a whole vector, except wo_Lazy which is called per chunk."""
    if not original or (method=='Yasuda' and len(inverse)!=len(original)):
        raise ValueError('Missing original or inverse ciphertext chunks')
    result=None
    for index,ct in enumerate(original):
        if method in {'ARMOR','wo_Lazy','wo_Ext'}:
            other=phantom.armor_rotate_without_key_switching(context,ct,0)
            product=phantom.armor_multiply(context,ct,other)
        elif method=='Yasuda':
            product=phantom.multiply(context,ct,inverse[index])
        elif method=='wo_Coeff':
            product=phantom.multiply(context,ct,ct)
        else:
            other=phantom.rotate(context,ct,0,material['galois_key'])
            product=phantom.multiply(context,ct,other)
        result=product if result is None else phantom.armor_add(context,result,product)
    if method in KEYSWITCH:
        result=phantom.relinearize(context,result,material['relin_key'])
    result=finish_product(context,result,spec)
    if method=='wo_Coeff':
        rotated=phantom.hoisting(context,result,material['galois_key'],list(range(1,16)))
        result=phantom.armor_add(context,result,rotated)
        for step in spec['galois_steps']:
            if step>=16:
                rotated=phantom.rotate(context,result,step,material['galois_key'])
                result=phantom.armor_add(context,result,rotated)
    return result


def partial_norm(method,context,spec,material,ciphertext,last):
    share=material['secret_share']
    if method in {'ARMOR','wo_Lazy','wo_Ext'}:
        operation=share.armor_individual_decrypt_last if last else share.armor_individual_decrypt
        partial=operation(context,ciphertext,material['product_share'],0,method!='wo_Ext',False)
    elif method=='Yasuda':
        # Both servers flood every coefficient before either partial is exposed.
        # False enables INTT + Karney DGS (sigma=2^52) over the full polynomial.
        partial=share.yasuda_individual_decrypt(context,ciphertext,material['product_share'],last,False,False)
    else:
        operation=share.armor_individual_decrypt_last if last else share.armor_individual_decrypt
        partial=operation(context,ciphertext,method!='wo_Coeff',True,False)
    if method in FULL_PARTIAL:
        return partial
    return phantom.armor_extract_partial_constant(context,partial,True,False)


def decode_norm(method,context,spec,material,local_partial,remote):
    if method in {'wo_Ext','Yasuda'}:
        remote=phantom.ciphertext.from_bytes(remote)
        # Flooded partials are in coefficient form. Combine and coefficient-
        # decode directly, without the NTT-only constant extraction routine
        # (which would also add a redundant third noise sample).
        plain=material['secret_share'].armor_multiparty_decrypt(context,[local_partial,remote])
        return float(material['encoder'].armor_coefficient_decode_and_mult(context,plain,1.)[0])
    if method=='wo_Coeff':
        remote=phantom.ciphertext.from_bytes(remote)
        plain=material['secret_share'].armor_multiparty_decrypt(context,[local_partial,remote])
        return float(material['encoder'].decode_double_vector(context,plain)[0])
    return float(phantom.armor_decode_constant(context,local_partial,remote,spec['scale'],1.))


def run_server2(method,local):
    args,context,spec,material=get_material(method,local,'server2')
    with listen(local.server2_host,local.server2_port) as listener:
        print(f'[{method}/Server2] listening {listener.address}',flush=True)
        with MeteredConnection(listener.accept(),args.log_path,'server2','server1') as conn:
            hello=expect(conn,'benchmark_hello')
            if hello['method']!=method:
                raise ValueError('Server1 method mismatch')
            while True:
                request=conn.recv()
                if request['type']=='stop':
                    break
                if request['type']!='norm_partial' or request['method']!=method:
                    raise ValueError('Unexpected norm request')
                ct=phantom.ciphertext.from_bytes(request['ciphertexts'][0])
                partial,elapsed=measured(partial_norm,method,context,spec,material,ct,True)
                payload=partial.to_bytes() if method in FULL_PARTIAL else partial
                conn.send(dict(type='norm_partial_result',round=request['round'],phase='norm',
                               job=request['job'],partial=payload,server2_compute_s=elapsed))
                log_event(args.log_path,'server2','norm_compute',round=request['round'],job=request['job'],latency_s=elapsed)


def norm_exchange(method,context,spec,material,conn,original,inverse,round_idx):
    he_s=rpc_s=local_s=server2_s=0.
    norm_squared=0.
    groups=[([ct],[]) for ct in original] if method in PER_CHUNK else [(original,inverse)]
    start=time.perf_counter()
    before=(conn.sent_bits,conn.received_bits)
    for job,(chunk_list,inverse_list) in enumerate(groups):
        product,elapsed=measured(compute_job,method,context,spec,material,chunk_list,inverse_list)
        he_s+=elapsed
        rpc_start=time.perf_counter()
        conn.send(dict(type='norm_partial',method=method,phase='norm',round=round_idx,
                       job=job,ciphertexts=[product.to_bytes()]))
        response=expect(conn,'norm_partial_result',round_idx)
        if response['job']!=job:
            raise ValueError('Norm partial job mismatch')
        rpc_s+=time.perf_counter()-rpc_start
        server2_s+=response['server2_compute_s']
        local_partial,elapsed=measured(partial_norm,method,context,spec,material,product,False)
        local_s+=elapsed
        value,elapsed=measured(decode_norm,method,context,spec,material,local_partial,response['partial'])
        local_s+=elapsed
        norm_squared+=value
    return dict(norm_squared=norm_squared,norm=math.sqrt(max(norm_squared,0)),
                he_s=he_s,rpc_s=rpc_s,local_decrypt_decode_s=local_s,server2_compute_s=server2_s,
                norm_total_s=time.perf_counter()-start,jobs=len(groups),
                s1_s2_tx_bytes=(conn.sent_bits-before[0])//8,
                s1_s2_rx_bytes=(conn.received_bits-before[1])//8)


def run_server1(method,local):
    args,context,spec,material=get_material(method,local,'server1')
    directory=configure_log(args,method,'server1')
    with MeteredConnection(open_peer(local.server2_host,local.server2_port,local.connect_timeout),args.log_path,'server1','server2') as peer:
        peer.send(dict(type='benchmark_hello',method=method))
        with listen(local.server1_host,local.server1_port) as listener:
            print(f'[{method}/Server1] waiting for client on {listener.address}',flush=True)
            with MeteredConnection(listener.accept(),args.log_path,'server1','client') as client_conn:
                hello=expect(client_conn,'benchmark_hello')
                if hello['method']!=method:
                    raise ValueError('Client method mismatch')
                round_idx=0
                failed=False
                for blocks in args.input_blocks:
                    rows=[]
                    for iteration in range(args.warmup+args.iterations):
                        for client_id in range(args.num_clients):
                            before=client_conn.received_bits
                            packet=expect(client_conn,'norm_input',round_idx)
                            upload_bytes=(client_conn.received_bits-before)//8
                            if packet['input_blocks']!=blocks or packet['elements']!=blocks*INPUT_BLOCK_SIZE:
                                raise ValueError('Wrong random input size/order')
                            original=unpack_ciphertexts(packet['ciphertexts'])
                            inverse=unpack_ciphertexts(packet.get('inverse_ciphertexts',[]))
                            width=args.ringdim//2 if method=='wo_Coeff' else args.ringdim
                            expected_chunks=(packet['elements']+width-1)//width
                            if len(original)!=expected_chunks or (method=='Yasuda' and len(inverse)!=expected_chunks):
                                raise ValueError('Incomplete random-input ciphertext upload')
                            client_conn.send(dict(type='input_received',round=round_idx))
                            result=norm_exchange(method,context,spec,material,peer,original,inverse,round_idx)
                            expected=packet['expected_norm_squared']
                            valid=math.isclose(result['norm_squared'],expected,rel_tol=args.rtol,abs_tol=args.atol)
                            error=abs(result['norm_squared']-expected)
                            row=dict(method=method,input_blocks=blocks,input_sha256=packet['input_sha256'],
                                     iteration=iteration,client_id=client_id,warmup=iteration<args.warmup,
                                     elements=packet['elements'],chunks=len(original),client_upload_bytes=upload_bytes,
                                     expected_norm_squared=expected,absolute_error=error if math.isfinite(error) else None,
                                     valid=valid,client_encrypt_s=packet['encrypt_s'],**result)
                            rows.append(row)
                            log_event(args.log_path,'server1','norm_result',round=round_idx,**row)
                            client_conn.send(dict(type='norm_result',round=round_idx,norm=result['norm'],valid=valid))
                            print(f'[{method}] blocks={blocks} elements={packet["elements"]} '
                                  f'iter={iteration} client={client_id} norm={result["norm"]:.6g} '
                                  f'valid={valid} HE={result["he_s"]*1000:.3f}ms '
                                  f'RPC={result["rpc_s"]*1000:.3f}ms upload={upload_bytes}B',flush=True)
                            round_idx+=1
                            del original,inverse
                    measured_rows=[r for r in rows if not r['warmup']]
                    summary=summarize_rows(measured_rows)
                    summary.update(method=method,input_blocks=blocks,input_block_size=INPUT_BLOCK_SIZE,
                                   input_distribution="normal(mean=0,std=0.01)",input_dtype="float64",
                                   input_sha256=measured_rows[0]['input_sha256'],spec=spec,
                                   elements=measured_rows[0]['elements'],chunks_per_client=measured_rows[0]['chunks'],
                                   chunk_width=width,padding_elements=measured_rows[0]['chunks']*width-measured_rows[0]['elements'],
                                   config={k:v for k,v in vars(args).items() if k!='log_path'},
                                   key_setup_scope='Shared once across all input sizes in this run',
                                   timing_note='RPC includes Server2 compute; norm_total already includes HE/RPC/local work',
                                   communication_note='Application frame bytes include 4/12-byte frame headers; exclude auth/TCP/IP/retransmits')
                    case_directory=directory/f'blocks_{blocks:04d}'
                    case_directory.mkdir(parents=True,exist_ok=True)
                    (case_directory/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
                    failed=failed or not summary['all_valid']
                peer.send(dict(type='stop'))
                if failed:
                    raise RuntimeError(f'{method}: numerical verification failed; see {directory}/blocks_*/summary.json')


def summarize_rows(rows):
    if not rows:
        raise ValueError('No measured rows')
    names=('client_upload_bytes','s1_s2_tx_bytes','s1_s2_rx_bytes','client_encrypt_s','he_s',
           'rpc_s','local_decrypt_decode_s','server2_compute_s','norm_total_s','jobs')
    errors=[r['absolute_error'] for r in rows if r['absolute_error'] is not None]
    result=dict(samples=len(rows),all_valid=all(r['valid'] for r in rows),
                max_absolute_error=max(errors) if errors else None)
    for name in names:
        values=np.array([r[name] for r in rows],dtype=np.float64)
        result[name]=dict(mean=float(values.mean()),std=float(values.std()),min=float(values.min()),max=float(values.max()),total=float(values.sum()))
    return result


def run_client(method,local):
    args,context,spec,material=get_material(method,local,'client')
    with MeteredConnection(open_peer(local.server1_host,local.server1_port,local.connect_timeout),args.log_path,'client','server1') as conn:
        conn.send(dict(type='benchmark_hello',method=method))
        round_idx=0
        for blocks in args.input_blocks:
            vector=input_vector(args,blocks)
            expected=float(vector.square().sum())
            digest=input_digest(vector)
            print(f'[{method}/Client] random input: {blocks}*16384={len(vector)}; sha256={digest}',flush=True)
            for iteration in range(args.warmup+args.iterations):
                for client_id in range(args.num_clients):
                    (original,inverse),elapsed=measured(encrypt_vector,method,context,spec,material,vector)
                    packet=dict(type='norm_input',round=round_idx,phase='norm',ciphertexts=pack_ciphertexts(original),
                                input_blocks=blocks,input_sha256=digest,
                                elements=len(vector),expected_norm_squared=expected,encrypt_s=elapsed)
                    if inverse:
                        packet['inverse_ciphertexts']=pack_ciphertexts(inverse)
                    start=time.perf_counter()
                    before=conn.sent_bits
                    conn.send(packet)
                    expect(conn,'input_received',round_idx)
                    log_event(args.log_path,'client','upload',round=round_idx,input_blocks=blocks,
                              elements=len(vector),input_sha256=digest,iteration=iteration,
                              warmup=iteration<args.warmup,client_id=client_id,encrypt_s=elapsed,
                              latency_s=time.perf_counter()-start,frame_bytes=(conn.sent_bits-before)//8)
                    result=expect(conn,'norm_result',round_idx)
                    if not result['valid']:
                        print(f'[{method}/Client] numerical mismatch round={round_idx}',flush=True)
                    round_idx+=1
                    del original,inverse,packet


def main(method,role,argv=None):
    if method not in METHODS or role not in ROLES:
        raise ValueError('Unknown method or role')
    args=parser(method).parse_args(argv)
    validate(args)
    configure_log(args,method,role)
    log_event(args.log_path,role,'benchmark_start',method=method,config=vars(args))
    if role=='keycenter':
        run_keycenter(method,args)
    elif role=='server1':
        run_server1(method,args)
    elif role=='server2':
        run_server2(method,args)
    else:
        run_client(method,args)
