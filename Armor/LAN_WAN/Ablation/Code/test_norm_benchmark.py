"""Algorithm/reference parity and norm measurement tests; CUDA tests use real HE."""
import contextlib
import io
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import torch
import norm_benchmark as bench
from summarize_norm_benchmarks import read_summary, main as summarize_main


class PlainCipher:
    def __init__(self,values,slots=False):
        self.values=np.asarray(values,dtype=np.float64)
        self.slots=slots
    def to_bytes(self):
        return json.dumps(dict(values=self.values.tolist(),slots=self.slots)).encode()


def from_bytes(blob):
    value=json.loads(blob)
    return PlainCipher(value['values'],value['slots'])


def inverse(values):
    return np.concatenate([values[:1],-values[:0:-1]])


def product(a,b):
    if a.slots:
        return PlainCipher(a.values*b.values,True)
    size=len(a.values)
    raw=np.convolve(a.values,b.values)
    result=raw[:size].copy()
    result[:size-1]-=raw[size:]
    return PlainCipher(result)


def add(a,b):
    return PlainCipher(a.values+b.values,a.slots)


def rotate(a,step):
    return PlainCipher(np.roll(a.values,-step),True) if a.slots else PlainCipher(inverse(a.values))


class AlgorithmTests(unittest.TestCase):
    def fake_native(self,stack):
        functions={
            'multiply':lambda c,a,b:product(a,b),
            'armor_multiply':lambda c,a,b:product(a,b),
            'armor_add':lambda c,a,b:add(a,b),
            'armor_rotate_without_key_switching':lambda c,a,s:rotate(a,s),
            'rotate':lambda c,a,s,k:rotate(a,s),
            'relinearize':lambda c,a,k:a,
            'armor_to_intt':lambda c,a:a,
            'armor_rescale_to':lambda c,a,l:a,
            'armor_to_ntt':lambda c,a:a,
            'hoisting':lambda c,a,k,steps:PlainCipher(sum(np.roll(a.values,-s) for s in steps),True)}
        return {name:stack.enter_context(patch.object(bench.phantom,name,side_effect=func))
                for name,func in functions.items()}

    def test_all_six_methods_compute_entire_vector_norm_and_reference_schedule(self):
        vector=np.linspace(-.4,.7,100)
        for method in bench.METHODS:
            with self.subTest(method=method),contextlib.ExitStack() as stack:
                mocks=self.fake_native(stack)
                spec=bench.specification(method,64)
                size=32 if method=='wo_Coeff' else 64
                padded=np.pad(vector,(0,(-len(vector))%size)).reshape(-1,size)
                cts=[PlainCipher(x,method=='wo_Coeff') for x in padded]
                inverses=[PlainCipher(inverse(x)) for x in padded] if method=='Yasuda' else []
                groups=[([ct],[]) for ct in cts] if method in bench.PER_CHUNK else [(cts,inverses)]
                total=0
                for originals,paired in groups:
                    result=bench.compute_job(method,None,spec,dict(relin_key='rk',galois_key='gk'),originals,paired)
                    total+=result.values[0]
                    if method=='wo_Coeff':
                        np.testing.assert_allclose(result.values,np.full(size,total))
                self.assertAlmostEqual(total,float(vector@vector),places=9)
                self.assertEqual(mocks['relinearize'].call_count,len(groups) if method in bench.KEYSWITCH else 0)
                self.assertEqual(mocks['armor_rescale_to'].call_count,len(groups))
                for call in mocks['armor_rescale_to'].call_args_list:
                    self.assertEqual(call.args[2],len(spec['modulus_chain'])-spec['size_P']-2)
                self.assertEqual(mocks['rotate'].call_count,len(cts)+(1 if method=='wo_Coeff' else 0)
                                 if method in {'wo_Dist'} else (1 if method=='wo_Coeff' else 0))
                self.assertEqual(mocks['armor_rotate_without_key_switching'].call_count,
                                 len(cts) if method in {'ARMOR','wo_Lazy','wo_Ext'} else 0)
                if method=='Yasuda':
                    mocks['armor_multiply'].assert_not_called()

    def test_key_packets_do_not_give_client_or_server2_evaluation_or_master_keys(self):
        for method in bench.METHODS:
            shares={'server1':(b's1',b'q1'),'server2':(b's2',b'q2')}
            evaluation={'relin_key':b'rk','galois_key':b'gk'} if method in bench.KEYSWITCH else {}
            client=bench.key_packet(method,'client',{}, {},b'pk',shares,evaluation)
            self.assertNotIn('secret_share',client)
            self.assertNotIn('product_share',client)
            self.assertNotIn('relin_key',client)
            s1=bench.key_packet(method,'server1',{}, {},b'pk',shares,evaluation)
            s2=bench.key_packet(method,'server2',{}, {},b'pk',shares,evaluation)
            self.assertEqual(s1['secret_share'],b's1')
            self.assertEqual(s2['secret_share'],b's2')
            self.assertNotIn('relin_key',s2)
            self.assertNotIn('galois_key',s2)
            self.assertEqual('product_share' in s1,method in {'ARMOR','wo_Lazy','wo_Ext','Yasuda'})

    def test_partial_routes_and_slot_full_coefficient_decoding(self):
        for method in bench.METHODS:
            for last in (False,True):
                share=Mock()
                material=dict(secret_share=share,product_share='product')
                with patch.object(bench.phantom,'armor_extract_partial_constant',return_value=[42]) as extract:
                    result=bench.partial_norm(method,'ctx',{},material,'ct',last)
                if method in {'ARMOR','wo_Lazy','wo_Ext'}:
                    operation=share.armor_individual_decrypt_last if last else share.armor_individual_decrypt
                    operation.assert_called_once_with('ctx','ct','product',0,method!='wo_Ext',False)
                elif method=='Yasuda':
                    share.yasuda_individual_decrypt.assert_called_once_with('ctx','ct','product',last,False,False)
                else:
                    operation=share.armor_individual_decrypt_last if last else share.armor_individual_decrypt
                    operation.assert_called_once_with('ctx','ct',method!='wo_Coeff',True,False)
                self.assertEqual(extract.call_count,0 if method in bench.FULL_PARTIAL else 1)
                if method=='Yasuda':
                    self.assertIs(result,share.yasuda_individual_decrypt.return_value)

    def test_yasuda_receives_and_combines_full_polynomials_before_extraction(self):
        local=PlainCipher([1.,9.,-7.])
        remote=PlainCipher([2.,-3.,5.])
        share=Mock()
        encoder=Mock()
        share.armor_multiparty_decrypt.side_effect=lambda c,parts:add(*parts)
        encoder.armor_coefficient_decode_and_mult.side_effect=lambda c,plain,scalar:plain.values/scalar
        with patch.object(bench.phantom,'ciphertext') as ciphertext, \
             patch.object(bench.phantom,'armor_extract_partial_constant') as extract:
            ciphertext.from_bytes.side_effect=from_bytes
            result=bench.decode_norm('Yasuda','ctx',{'scale':88},
                                     dict(secret_share=share,encoder=encoder),local,remote.to_bytes())
        self.assertEqual(result,3.)
        share.armor_multiparty_decrypt.assert_called_once()
        args=encoder.armor_coefficient_decode_and_mult.call_args.args
        self.assertEqual(args[0],'ctx')
        np.testing.assert_allclose(args[1].values,[3.,6.,-2.])
        self.assertEqual(args[2],1.)
        extract.assert_not_called()

    def test_yasuda_server2_sends_entire_partial_polynomial(self):
        args=SimpleNamespace(server2_host='local',server2_port=2,log_path=None)
        polynomial=PlainCipher([3.,11.,-5.,8.])
        raw=Mock()
        requests=[dict(type='benchmark_hello',method='Yasuda'),
                  dict(type='norm_partial',method='Yasuda',round=0,job=0,ciphertexts=[b'ct']),
                  dict(type='stop')]
        raw.recv_bytes.side_effect=[pickle.dumps(x) for x in requests]
        listener=MagicMock()
        listener.__enter__.return_value=listener
        listener.accept.return_value=raw
        with patch.object(bench,'get_material',return_value=(args,None,{},{})), \
             patch.object(bench,'listen',return_value=listener), \
             patch.object(bench.phantom,'ciphertext'), \
             patch.object(bench,'measured',return_value=(polynomial,.001)), \
             contextlib.redirect_stdout(io.StringIO()):
            bench.run_server2('Yasuda',args)
        response=pickle.loads(raw.send_bytes.call_args.args[0])
        self.assertIsInstance(response['partial'],bytes)
        np.testing.assert_array_equal(from_bytes(response['partial']).values,polynomial.values)

    def test_m3_receives_and_combines_full_polynomials_before_extraction(self):
        local=PlainCipher([1.,9.,-7.])
        remote=PlainCipher([2.,-3.,5.])
        share=Mock()
        encoder=Mock()
        share.armor_multiparty_decrypt.side_effect=lambda c,parts:add(*parts)
        encoder.armor_coefficient_decode_and_mult.side_effect=lambda c,plain,scalar:plain.values/scalar
        with patch.object(bench.phantom,'ciphertext') as ciphertext, \
             patch.object(bench.phantom,'armor_extract_partial_constant') as extract:
            ciphertext.from_bytes.side_effect=from_bytes
            result=bench.decode_norm('wo_Ext','ctx',{'scale':88},
                                     dict(secret_share=share,encoder=encoder),local,remote.to_bytes())
        self.assertEqual(result,3.)
        share.armor_multiparty_decrypt.assert_called_once()
        args=encoder.armor_coefficient_decode_and_mult.call_args.args
        self.assertEqual(args[0],'ctx')
        np.testing.assert_allclose(args[1].values,[3.,6.,-2.])
        self.assertEqual(args[2],1.)
        extract.assert_not_called()

    def test_m3_server2_sends_entire_partial_polynomial(self):
        args=SimpleNamespace(server2_host='local',server2_port=2,log_path=None)
        polynomial=PlainCipher([3.,11.,-5.,8.])
        raw=Mock()
        requests=[dict(type='benchmark_hello',method='wo_Ext'),
                  dict(type='norm_partial',method='wo_Ext',round=0,job=0,ciphertexts=[b'ct']),
                  dict(type='stop')]
        raw.recv_bytes.side_effect=[pickle.dumps(x) for x in requests]
        listener=MagicMock()
        listener.__enter__.return_value=listener
        listener.accept.return_value=raw
        with patch.object(bench,'get_material',return_value=(args,None,{},{})), \
             patch.object(bench,'listen',return_value=listener), \
             patch.object(bench.phantom,'ciphertext'), \
             patch.object(bench,'measured',return_value=(polynomial,.001)), \
             contextlib.redirect_stdout(io.StringIO()):
            bench.run_server2('wo_Ext',args)
        response=pickle.loads(raw.send_bytes.call_args.args[0])
        self.assertIsInstance(response['partial'],bytes)
        np.testing.assert_array_equal(from_bytes(response['partial']).values,polynomial.values)

    def test_measured_rows_exclude_warmup_and_summary_does_not_add_mirrored_bytes(self):
        rows=[]
        for i in range(2):
            row={name:2+i for name in ('client_upload_bytes','s1_s2_tx_bytes','s1_s2_rx_bytes',
                'client_encrypt_s','he_s','rpc_s','local_decrypt_decode_s','server2_compute_s','norm_total_s','jobs')}
            row.update(valid=True,absolute_error=.001)
            rows.append(row)
        summary=bench.summarize_rows(rows)
        self.assertEqual(summary['samples'],2)
        self.assertEqual(summary['he_s']['mean'],2.5)
        summary.update(method='wo_Dist',config=dict(run_tag='lan_rep1'))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'summary.json'
            path.write_text(json.dumps(summary))
            (path.parent/'keycenter.jsonl').write_text(json.dumps(dict(event='communication',direction='send',frame_bits=800))+'\n')
            result=read_summary(path)
            self.assertEqual(result['s1_s2_total_bytes'],5)
            self.assertEqual(result['keycenter_tx_bytes'],100)
            self.assertEqual(result['he_ms'],2500)

    def test_actual_frame_metering_and_rpc_count_for_each_method(self):
        vector=np.linspace(-.4,.7,100)
        for method in bench.METHODS:
            with self.subTest(method=method),contextlib.ExitStack() as stack:
                self.fake_native(stack)
                stack.enter_context(patch.object(bench.phantom,'cudaStreamSychronize'))
                spec=bench.specification(method,64)
                width=32 if method=='wo_Coeff' else 64
                padded=np.pad(vector,(0,(-len(vector))%width)).reshape(-1,width)
                cts=[PlainCipher(x,method=='wo_Coeff') for x in padded]
                paired=[PlainCipher(inverse(x)) for x in padded] if method=='Yasuda' else []
                class Raw:
                    def __init__(self):
                        self.sent=[];self.responses=[]
                    def send_bytes(self,payload):
                        self.sent.append(payload)
                        request=pickle.loads(payload)
                        value=from_bytes(request['ciphertexts'][0])
                        partial=(PlainCipher(value.values*.6,value.slots).to_bytes() if method in bench.FULL_PARTIAL
                                 else [float(value.values[0]*.6)])
                        self.response=pickle.dumps(dict(type='norm_partial_result',round=request['round'],
                            phase='norm',job=request['job'],partial=partial,server2_compute_s=.0001),protocol=pickle.HIGHEST_PROTOCOL)
                    def recv_bytes(self):
                        self.responses.append(self.response)
                        return self.response
                def partial(m,c,s,keys,ct,last):
                    return PlainCipher(ct.values*.4,ct.slots) if m in bench.FULL_PARTIAL else [ct.values[0]*.4]
                def decode(m,c,s,keys,own,remote):
                    return float(own.values[0]+from_bytes(remote).values[0]) if m in bench.FULL_PARTIAL else own[0]+remote[0]
                stack.enter_context(patch.object(bench,'partial_norm',side_effect=partial))
                stack.enter_context(patch.object(bench,'decode_norm',side_effect=decode))
                raw=Raw()
                conn=bench.MeteredConnection(raw,None,'server1','server2')
                result=bench.norm_exchange(method,None,spec,dict(relin_key='rk',galois_key='gk'),conn,cts,paired,0)
                self.assertAlmostEqual(result['norm_squared'],float(vector@vector),places=9)
                self.assertEqual(result['jobs'],len(cts) if method in bench.PER_CHUNK else 1)
                self.assertEqual(result['s1_s2_tx_bytes'],sum(len(x)+4 for x in raw.sent))
                self.assertEqual(result['s1_s2_rx_bytes'],sum(len(x)+4 for x in raw.responses))

    def test_server1_sweeps_exact_sizes_and_excludes_warmup_per_size(self):
        with tempfile.TemporaryDirectory() as directory:
            args=SimpleNamespace(output=directory,run_tag='test',gpu=0,ringdim=16384,warmup=1,
                iterations=1,num_clients=2,rtol=.001,atol=.001,input_blocks=[62,184,367],server2_host='local',server2_port=2,
                server1_host='local',server1_port=1,connect_timeout=1)
            bench.configure_log(args,'ARMOR','server1')
            raw=Mock()
            messages=[dict(type='benchmark_hello',method='ARMOR')]
            round_idx=0
            for blocks in args.input_blocks:
                for _ in range(4):
                    messages.append(dict(type='norm_input',round=round_idx,ciphertexts=[b'ct']*blocks,
                        input_blocks=blocks,input_sha256=f'hash{blocks}',elements=blocks*16384,
                        expected_norm_squared=9.,encrypt_s=.1))
                    round_idx+=1
            raw.recv_bytes.side_effect=[pickle.dumps(x) for x in messages]
            listener=MagicMock()
            listener.__enter__.return_value=listener
            listener.accept.return_value=raw
            result=dict(norm_squared=9.,norm=3.,he_s=.1,rpc_s=.2,local_decrypt_decode_s=.3,
                server2_compute_s=.1,norm_total_s=.6,jobs=1,s1_s2_tx_bytes=100,s1_s2_rx_bytes=20)
            with patch.object(bench,'get_material',return_value=(args,None,{},{})), \
                 patch.object(bench,'open_peer',return_value=Mock()), \
                 patch.object(bench,'listen',return_value=listener), \
                 patch.object(bench,'unpack_ciphertexts',side_effect=lambda cts:[None]*len(cts)), \
                 patch.object(bench,'norm_exchange',return_value=result) as exchange, \
                 contextlib.redirect_stdout(io.StringIO()):
                bench.run_server1('ARMOR',args)
            self.assertEqual(exchange.call_count,12)
            for blocks in args.input_blocks:
                path=Path(directory)/f'ARMOR/test/blocks_{blocks:04d}/summary.json'
                summary=json.loads(path.read_text())
                self.assertEqual(summary['samples'],2)
                self.assertEqual(summary['he_s']['total'],.2)
                self.assertEqual(summary['elements'],blocks*16384)
                self.assertEqual(summary['chunks_per_client'],blocks)
                self.assertEqual(read_summary(path)['input_blocks'],blocks)
            rows=[json.loads(x) for x in Path(args.log_path).read_text().splitlines()]
            results=[r for r in rows if r['event']=='norm_result']
            self.assertEqual([r['warmup'] for r in results],[True,True,False,False]*3)
            # Old model-run summaries must not be mixed into the new sweep CSV.
            (Path(directory)/'ARMOR/test/summary.json').write_text('old model output')
            csv_path=Path(directory)/'comparison.csv'
            output=io.StringIO()
            with patch('sys.argv',['summary',directory,'--csv',str(csv_path)]),contextlib.redirect_stdout(output):
                summarize_main()
            self.assertEqual(len(csv_path.read_text().splitlines()),4)
            self.assertIn('6012928',output.getvalue())

    def test_large_ring_padding_preserves_input_and_norm(self):
        for degree in (32768,65536):
            for width in (degree,degree//2):
                for blocks in bench.INPUT_BLOCKS:
                    vector=bench.input_vector(SimpleNamespace(seed=2023),blocks)
                    chunks=bench.encrypt.dividing_by_chunk(vector,SimpleNamespace(ringdim=width))
                    joined=torch.cat(chunks)
                    self.assertEqual(len(chunks),(vector.numel()+width-1)//width)
                    self.assertTrue(torch.equal(joined[:vector.numel()],vector))
                    self.assertEqual(torch.count_nonzero(joined[vector.numel():]).item(),0)
                    self.assertAlmostEqual(float(joined.square().sum()),float(vector.square().sum()),places=8)

    def test_random_inputs_have_exact_requested_lengths_and_repeatable_hashes(self):
        args=SimpleNamespace(seed=2023)
        for blocks,size in ((62,1015808),(184,3014656),(367,6012928)):
            a=bench.input_vector(args,blocks)
            b=bench.input_vector(args,blocks)
            self.assertEqual(a.numel(),size)
            self.assertEqual(a.dtype,torch.float64)
            self.assertEqual(bench.input_digest(a),bench.input_digest(b))
            self.assertTrue(torch.isfinite(a).all())
            self.assertGreater(float(a.std()),0)
        defaults=bench.parser('ARMOR').parse_args([])
        self.assertEqual(defaults.input_blocks,[62,184,367])
        self.assertFalse(hasattr(defaults,'model'))
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            bench.parser('ARMOR').parse_args(['--model','resnet34'])

    def test_bad_message_and_no_input_fail_explicitly(self):
        conn=Mock()
        conn.recv.return_value=dict(type='wrong',round=0)
        with self.assertRaises(ValueError):
            bench.expect(conn,'norm_input',0)
        for method in bench.METHODS:
            with self.assertRaises(ValueError):
                bench.compute_job(method,None,{}, {},[],[])


@unittest.skipUnless(torch.cuda.is_available(),'Requires a working CUDA GPU')
class NativeNormTests(unittest.TestCase):
    def test_each_method_actual_serialized_eval_keys_partials_and_norm(self):
        for method in bench.METHODS:
            with self.subTest(method=method):
                spec=bench.specification(method,16384)
                context=bench.make_context(spec)
                master=bench.phantom.secret_key(context)
                s1,s2,q1,q2=bench.generate_key_shares(master,context,method)
                material=dict(public_key=master.gen_publickey(context),encoder=bench.phantom.ckks_encoder(context),
                              secret_share=bench.phantom.secret_key.from_bytes(s1.to_bytes()),
                              product_share=bench.phantom.secret_key.from_bytes(q1.to_bytes()))
                remote=dict(secret_share=bench.phantom.secret_key.from_bytes(s2.to_bytes()),
                            product_share=bench.phantom.secret_key.from_bytes(q2.to_bytes()))
                if method in bench.KEYSWITCH:
                    material['relin_key']=bench.phantom.relin_key.from_bytes(master.gen_relinkey(context).to_bytes())
                    material['galois_key']=bench.phantom.galois_key.from_bytes(master.create_galois_keys(context).to_bytes())
                vector=torch.randn(16387,dtype=torch.float64)*.01
                original,inverse_ct=bench.encrypt_vector(method,context,spec,material,vector)
                original=bench.unpack_ciphertexts(bench.pack_ciphertexts(original))
                inverse_ct=bench.unpack_ciphertexts(bench.pack_ciphertexts(inverse_ct))
                groups=[([ct],[]) for ct in original] if method in bench.PER_CHUNK else [(original,inverse_ct)]
                total=0
                for cts,inverses in groups:
                    product_ct=bench.compute_job(method,context,spec,material,cts,inverses)
                    product_ct=bench.phantom.ciphertext.from_bytes(product_ct.to_bytes())
                    local=bench.partial_norm(method,context,spec,material,product_ct,False)
                    partial=bench.partial_norm(method,context,spec,remote,product_ct,True)
                    if method in bench.FULL_PARTIAL:
                        partial=partial.to_bytes()
                    total+=bench.decode_norm(method,context,spec,material,local,partial)
                np.testing.assert_allclose(total,float(vector.square().sum()),rtol=1e-3,atol=1e-3)


if __name__=='__main__':
    unittest.main()
