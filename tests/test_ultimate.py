"""CPU-only correctness gates for new scientific execution paths."""
import copy
import tempfile
from pathlib import Path
import unittest

import chess

from execution.support import atomic_json, fingerprint
from execution.ultimate_contract import (CONTRACT, DEADLINE, deadline_check, freeze, group_id,
                                         group_variants, hard_ids, safe_output, validate_observation)
from execution.ultimate_data import validate_openings
from execution.ultimate_match import (assignments, make_envelope, play, summary, timing_gate,
                                      validate_game)
from execution.ultimate_study import conditions, observation, read_block


class FakeEngine:
    def __init__(self, wall=.08, bad=False):
        self.calls, self.wall, self.bad = [], wall, bad

    def newgame(self):
        self.calls.append('reset')

    def go(self, **kwargs):
        self.calls.append(kwargs)
        board = chess.Board(kwargs.get('fen', chess.STARTING_FEN))
        for move in kwargs.get('moves', []):
            board.push_uci(move)
        return {'bestmove': 'a1a8' if self.bad else next(iter(board.legal_moves)).uci(),
                'wall': self.wall, 'nodes': 100, 'time_ms': 80, 'score_cp': 0, 'mate': None}


class UltimateTests(unittest.TestCase):
    opening = ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6']

    def test_extracted_protocol_json_roundtrip_preserves_contract(self):
        import json
        restored=json.loads(json.dumps(CONTRACT))
        self.assertEqual(fingerprint(restored),fingerprint(CONTRACT))
        restored['timed']['MoveOverheadMs']=50
        self.assertNotEqual(fingerprint(restored),fingerprint(CONTRACT))

    def test_backup_gate_requires_U10_and_full_replay_requires_backup(self):
        from unittest.mock import patch
        from execution.ultimate import Run
        with tempfile.TemporaryDirectory() as temp:
            run=object.__new__(Run)
            run.fresh,run.recovery,run.cfg=True,None,{}
            run.out=Path(temp);run.state_path=run.out/'state.json'
            for phase,previous in (('U10A','U10'),('U11','U10A')):
                run.state={'phases':{previous:{'status':'awaiting_agent_review'}}}
                with patch('execution.ultimate.working_branch'),patch('execution.ultimate.deadline_check'):
                    with self.assertRaisesRegex(RuntimeError,'Previous phase not verified'):
                        run.phase(phase)

    def root(self):
        return {'id': 1, 'fen': chess.STARTING_FEN,
                'moves': {m.uci(): {'res': 0} for m in chess.Board().legal_moves},
                'correct': ['e2e4']}

    def test_agent_review_resume_needs_no_failure_recovery_override(self):
        from contextlib import nullcontext, redirect_stdout
        import io
        from unittest.mock import patch
        from execution.ultimate import Run
        for reviewed in (False, True):
            with self.subTest(reviewed=reviewed), tempfile.TemporaryDirectory() as temp:
                run = object.__new__(Run)
                run.fresh, run.recovery, run.cfg = True, None, {}
                run.out = Path(temp)
                run.state_path = run.out/'state.json'
                run.state = {'phases': {'U7': {'status': 'verified'},
                                       'U8': {'status': 'awaiting_agent_review', 'outputs': []}}}
                artifact = run.out/'checked.txt'
                artifact.write_text('review fixture', encoding='utf-8')
                result = {'outputs': [artifact]}
                if not reviewed:
                    result['awaiting_agent_review'] = ['plot.png']
                with patch('execution.ultimate.log'), patch('execution.ultimate.working_branch'), \
                     patch('execution.ultimate.deadline_check'), \
                     patch('execution.ultimate_study.keep_awake', return_value=nullcontext()), \
                     patch('execution.ultimate_phases.U8', return_value=result) as phase, \
                     redirect_stdout(io.StringIO()):
                    returned = run.phase('U8')
                phase.assert_called_once_with(run)
                saved = run.state['phases']['U8']
                self.assertEqual(saved['status'], 'verified' if reviewed else 'awaiting_agent_review')
                self.assertEqual(saved['attempt_history'][0]['status'], 'awaiting_agent_review')
                self.assertEqual('completed_IST' in saved, reviewed)
                if not reviewed:
                    self.assertIs(returned, False)
                # A real failure still requires a diagnosed recovery note.
                run.state['phases']['U8'] = {'status': 'BLOCKED'}
                with patch('execution.ultimate.working_branch'), patch('execution.ultimate.deadline_check'):
                    with self.assertRaisesRegex(RuntimeError, 'diagnosed recovery'):
                        run.phase('U8')

    def test_exact_budget_and_legal_output(self):
        r = self.root()
        row = {'id': 1, 'budget': 100, 'best': 'e2e4', 'nodes': 257, 'time_ms': 20, 'wall': .025}
        validate_observation(row, r, 'B', 100)
        for change in ({'budget': 400}, {'id': 2}, {'best': 'a1a8'}, {'nodes': None},
                       {'time_ms': 0}, {'wall': float('nan')}):
            with self.assertRaises(ValueError):
                validate_observation({**row, **change}, r, 'B', 100)

    def test_raw_keeps_imported_best_not_policy_argmax(self):
        r = self.root()
        res = {'bestmove': 'e2e4', 'nodes': 1, 'wl': .2, 'd': .4, 'v': .2, 'm': 30,
               'moves': {m: {'p': .9 if m == 'd2d4' else .01} for m in r['moves']}}
        row = observation(r, res, 'A', 0)
        self.assertEqual(row['best'], 'e2e4')
        self.assertEqual(max(row['p'], key=row['p'].get), 'd2d4')
        with self.assertRaises(ValueError):
            validate_observation({**row, 'p': {}}, r, 'A', 0)

    def test_frozen_challenge_rule_and_denominator(self):
        roots = [{'id': 1, 'correct': ['a']}, {'id': 2, 'correct': ['a']}, {'id': 3, 'correct': ['a']}]
        raw = {1: {'best': 'a', 'p': {'a': .8}}, 2: {'best': 'a', 'p': {'a': .7999}},
               3: {'best': 'b', 'p': {'a': .9}}}
        self.assertEqual(hard_ids(roots, raw), [2, 3])
        with self.assertRaises(ValueError):
            hard_ids(roots, {1: raw[1]})

    def test_legal_symmetry_and_color_mirror(self):
        b = chess.Board('8/8/5k2/8/2P5/4K3/8/8 w - - 0 1')
        variants = group_variants(b)
        self.assertEqual(len(variants), 4)
        self.assertTrue(all(x.is_valid() for x in variants))
        self.assertEqual({group_id(x) for x in variants}, {group_id(b)})
        pawnless = chess.Board('8/8/5k2/8/2R5/4K3/8/8 w - - 0 1')
        self.assertEqual(len(group_variants(pawnless)), 16)

    def test_no_input_or_final_output_collision(self):
        from execution.ultimate_contract import ROOT
        for path in (ROOT, ROOT/'models', ROOT/'data/sub', ROOT/'submissionready/new'):
            with self.assertRaises(ValueError):
                safe_output(path)
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(safe_output(temp), Path(temp).resolve())

    def test_missing_or_corrupt_resume_output_is_not_accuracy_zero(self):
        r = self.root()
        contract = {'study': 'B', 'budget': 100}
        with tempfile.TemporaryDirectory() as temp:
            self.assertIsNone(read_block(temp, contract, [r]))
            path = Path(temp)/'output.json'
            atomic_json(path, {'status': 'verified', 'contract': fingerprint(contract), 'rows': []})
            with self.assertRaises(ValueError):
                read_block(temp, contract, [r])
            path.write_text('{broken', encoding='utf-8')
            with self.assertRaises(ValueError):
                read_block(temp, contract, [r])

    def test_resume_contract_change_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/'contract.json'
            freeze(p, {'model': 's4', 'budget': 100})
            freeze(p, {'model': 's4', 'budget': 100})
            with self.assertRaises(ValueError):
                freeze(p, {'model': 's4', 'budget': 200})

    @unittest.skipUnless(__import__('os').name=='nt','Native Windows rename sharing test')
    def test_atomic_checkpoint_survives_transient_windows_reader(self):
        import ctypes
        from ctypes import wintypes
        import threading
        import time
        from execution.ultimate_contract import atomic_json as commit
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'checkpoint.json'
            p.write_bytes(b'{"old": true}\n')
            kernel=ctypes.WinDLL('kernel32',use_last_error=True)
            kernel.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,
                                        wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
            kernel.CreateFileW.restype=wintypes.HANDLE
            kernel.CloseHandle.argtypes=[wintypes.HANDLE]
            handle=kernel.CreateFileW(str(p),0x80000000,1,None,3,0,None)
            self.assertNotEqual(handle,ctypes.c_void_p(-1).value)
            def release():
                time.sleep(.2)
                kernel.CloseHandle(handle)
            thread=threading.Thread(target=release)
            thread.start()
            try:
                commit(p,{'new':True})
            finally:
                thread.join()
            self.assertEqual(p.read_bytes(),b'{\n  "new": true\n}\n')

    @unittest.skipUnless(__import__('os').name=='nt','Windows-specific bounded rename recovery')
    def test_persistent_atomic_error_keeps_last_good_and_complete_candidate(self):
        from unittest.mock import patch
        from execution.ultimate_contract import atomic_json as commit
        import json
        failure=PermissionError('persistent sharing denial')
        failure.winerror=5
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'checkpoint.json'
            p.write_bytes(b'{"old": true}\n')
            with patch('execution.ultimate_contract.os.replace',side_effect=failure) as replace, \
                    patch('execution.ultimate_contract.time.sleep'):
                with self.assertRaises(PermissionError):
                    commit(p,{'new':True})
            self.assertEqual(replace.call_count,7)
            self.assertEqual(p.read_bytes(),b'{"old": true}\n')
            candidate=Path(replace.call_args.args[0])
            try:
                self.assertEqual(json.loads(candidate.read_bytes()),{'new':True})
            finally:
                candidate.unlink()

    def test_deadline_protects_buffer(self):
        deadline_check(10, now=DEADLINE-100, deadline=DEADLINE)
        with self.assertRaises(TimeoutError):
            deadline_check(10, now=DEADLINE-65, deadline=DEADLINE)

    def test_background_flags_informational_but_real_failures_block(self):
        from types import SimpleNamespace
        good = {'ac_line': 1, 'gpu': dict(temp_c=40, sm_mhz=2000, mem_mhz=8000, util=30, power_w=40)}
        conditions(SimpleNamespace(errors=[], samples=[good]))
        with self.assertRaises(RuntimeError):
            conditions(SimpleNamespace(errors=['monitor failed'], samples=[good]))
        with self.assertRaises(RuntimeError):
            conditions(SimpleNamespace(errors=[], samples=[{**good, 'ac_line': 0}]))

    def test_book_terminal_uniqueness(self):
        self.assertEqual(validate_openings([self.opening, self.opening], 1), [self.opening])
        with self.assertRaises(ValueError):
            validate_openings([self.opening[:4]])

    def test_pair_order_counts_and_interleaving(self):
        keys = list(assignments())
        self.assertEqual(len(keys), 1000)
        self.assertEqual(len(set(keys)), 1000)
        for budget in (100, 200):
            self.assertEqual({(p, c) for b, p, c in keys if b == budget},
                             {(p, c) for p in range(250) for c in (0, 1)})
        self.assertEqual(keys[10][0], 200)
        self.assertEqual(keys[2:4], [(100, 1, 1), (100, 1, 0)])

    def test_fixed_movetime_no_fischer_clock_and_envelope(self):
        # Bounded mock fixture; real contract remains 400 plies.
        from unittest.mock import patch
        settings = {**CONTRACT['match'], 'max_plies': 10}
        a, b = FakeEngine(), FakeEngine()
        with patch.dict(CONTRACT, {'match': settings}):
            board, result, reason, stats = play(a, b, self.opening, 100, '')
        self.assertEqual(reason, 'max_plies')
        calls = [x for e in (a, b) for x in e.calls if isinstance(x, dict)]
        self.assertTrue(calls)
        self.assertTrue(all(x['movetime'] == 100 and x['timeout'] == 30 and 'clock' not in x for x in calls))
        contract = {'openings': [self.opening]}
        env = make_envelope(board, result, reason, stats, 100, 0, 0, 1, contract)
        validate_game(env, contract)
        changed = copy.deepcopy(env)
        changed['record']['color'] = 1
        with self.assertRaises(ValueError):
            validate_game(changed, contract)
        self.assertEqual(summary([env])[0]['draws'], 1)

    def test_illegal_match_move_is_pipeline_failure(self):
        with self.assertRaises(ValueError):
            play(FakeEngine(bad=True), FakeEngine(), self.opening, 100, '')

    def test_pilot_timing_gate_checks_both_models_budgets(self):
        envs = []
        for budget in (100, 200):
            envs.append({'record': {'budget_ms': budget, 'white': 'fp16', 'black': 's4',
                                   'moves': [{'side': s, 'wall_ms': budget} for s in ('white', 'black')]}})
        self.assertEqual(timing_gate(envs)['status'], 'verified')
        envs[0]['record']['moves'][0]['wall_ms'] = 150
        self.assertEqual(timing_gate(envs)['status'], 'BLOCKED')

    def test_match_counters_keep_zero_work_but_reject_missing_negative(self):
        from execution.ultimate_match import match_counters
        match_counters(0,0,.1)
        for args in ((None,1,1),(1,None,1),(-1,1,1),(1,-1,1),(1,1,0),(1.5,1,1)):
            with self.assertRaises(ValueError):
                match_counters(*args)

    def test_package_manifest_detects_changed_input_and_zip_traversal(self):
        import zipfile
        from execution.ultimate_package import verify_manifest,extract_verified,archive,deadline_copy,NAME
        from execution.ultimate_contract import portable_identity
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);folder=base/NAME;folder.mkdir()
            source=folder/'input.dat';source.write_bytes(b'canonical')
            atomic_json(folder/'artifact-manifest.json',{'files':[portable_identity(source,folder)]})
            verify_manifest(folder)
            good=archive(folder,base/'valid.zip')
            extracted=extract_verified(good,base/'valid-extraction')
            self.assertEqual((extracted/'input.dat').read_bytes(),b'canonical')
            self.assertEqual((extracted/'artifact-manifest.json').read_bytes(),(folder/'artifact-manifest.json').read_bytes())
            copied=base/'deadline-copy.dat'
            deadline_copy(source,copied)
            self.assertEqual(copied.read_bytes(),source.read_bytes())
            with self.assertRaises(FileExistsError):deadline_copy(source,copied)
            source.write_bytes(b'changed')
            with self.assertRaises(ValueError):verify_manifest(folder)
            zipped=base/'bad.zip'
            with zipfile.ZipFile(zipped,'w') as stream:stream.writestr(NAME+'/../outside.txt','bad')
            with self.assertRaises(ValueError):extract_verified(zipped,base/'extraction')
            self.assertFalse((base/'outside.txt').exists())

    def test_fresh_native_copy_excludes_old_caches_and_binds_actual_bytes(self):
        from execution.ultimate_replay import prepare_runtime
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);native=base/'installed';native.mkdir()
            executable=native/'lc0.exe';executable.write_bytes(b'actual pinned executable')
            dll=native/'runtime.dll';dll.write_bytes(b'actual pinned DLL')
            (native/'trt_cache').mkdir();(native/'trt_cache/old.engine').write_bytes(b'old cached tactics')
            cfg={'lc0':str(executable)};out=base/'replay'
            copied=prepare_runtime(cfg,out)
            self.assertEqual(Path(copied['lc0']).read_bytes(),executable.read_bytes())
            self.assertFalse((out/'runtime/lc0/trt_cache').exists())
            prepare_runtime(cfg,out)
            (out/'runtime/lc0/runtime.dll').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):prepare_runtime(cfg,out)

    def test_bootstrap_clusters_and_flat_reference_intervals(self):
        from analysis.ultimate import bootstrap,node_interval,holm
        roots=[{'cls':'KPvK','fen':'8/8/5k2/8/2P5/4K3/8/8 w - - 0 1'},
               {'cls':'KPvK','fen':'8/8/2k5/8/5P2/3K4/8/8 w - - 0 1'}]
        self.assertEqual(bootstrap(roots,[1,1],replicates=100),[[1.,1.]])
        # The two legal reflections form one cluster: rootwise sampling would
        # incorrectly vary this mean; joint endpoints preserve the same pairs.
        self.assertEqual(bootstrap(roots,[0,1],replicates=100),[[.5,.5]])
        self.assertEqual(bootstrap(roots,[[0,1],[1,0]],replicates=100),[[.5,.5],[.5,.5]])
        interval=node_interval(1,[(1,.9),(100,1),(400,1),(800,1)])
        self.assertAlmostEqual(interval[0],100)
        self.assertIsNone(interval[1])
        rows=[dict(cohort='M',study='B',model='s4',nominal_mcnemar_p=.01),
              dict(cohort='M',study='B',model='s1',nominal_mcnemar_p=.04),
              dict(cohort='HN',study='B',model='s4',nominal_mcnemar_p=.01),
              dict(cohort='HN',study='B_s0',model='s0',nominal_mcnemar_p=.04)]
        holm(rows)
        self.assertEqual([r['holm_p'] for r in rows],[.02,.04,.02,.04])
        self.assertEqual(rows[2]['holm_family'],rows[3]['holm_family'])


if __name__ == '__main__':
    unittest.main()
