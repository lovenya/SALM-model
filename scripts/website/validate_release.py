#!/usr/bin/env python3
"""Validate a static paper companion and its exact curated media without inference."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import struct

class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.links = [], []
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.append(attrs['id'])
        if tag in {'a', 'script', 'link'}:
            value = attrs.get('src', attrs.get('href', ''))
            if value and not value.startswith(('https:', 'http:', 'mailto:', '#')):
                self.links.append(value.split('?')[0].split('#')[0])

def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def check_wav(path):
    with path.open('rb') as handle:
        assert handle.read(4) == b'RIFF', path
        handle.read(4)
        assert handle.read(4) == b'WAVE', path
        fmt, data_size = None, None
        while header := handle.read(8):
            assert len(header) == 8, path
            name, size = struct.unpack('<4sI', header)
            if name == b'fmt ':
                payload = handle.read(size)
                fmt = struct.unpack('<HHIIHH', payload[:16])
            elif name == b'data':
                data_size = size
                assert handle.tell() + size <= path.stat().st_size, path
                handle.seek(size, 1)
            else:
                handle.seek(size, 1)
            if size % 2:
                handle.seek(1, 1)
        assert fmt and fmt[0] in (1, 3, 65534) and fmt[1] > 0 and fmt[2] > 0, path
        assert data_size and data_size > 0, path
        return fmt[2]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('release', type=Path)
    parser.add_argument('--survey', type=Path, default=Path('/scratch/lovenya/salm_model_outputs/user_survey_bundle_20260917'))
    parser.add_argument('--eval-repo', type=Path, default=Path('/scratch/lovenya/SpeechAudioLM'))
    args = parser.parse_args()
    root = args.release.resolve()
    raw = (root / 'site-data.js').read_text()
    data = json.loads(raw.removeprefix('window.PAPER_DATA = ').removesuffix(';\n'))
    manifest = json.loads((root / 'build-manifest.json').read_text())
    assert data['release'] is True and (root / '.nojekyll').is_file()
    assert '/scratch/' not in raw and '/home/' not in raw, 'Private filesystem paths in public data'
    samples = data['samples']
    assert len(samples) == len({s['sample_id'] for s in samples}) == manifest['examples']
    originals = {row['sample_id']: row for row in map(json.loads, (args.survey / 'samples.jsonl').read_text().splitlines())}
    rationales = {}
    for direction in ('a2s', 's2a'):
        manifest_path = args.eval_repo / f'data/manifests/clean_v2/{direction}/ttas_test.jsonl'
        rationales[direction] = {Path(row['audio_file']).stem: row['reasoning'] for row in map(json.loads, manifest_path.read_text().splitlines())}
    geval = data['geval']
    assert geval['promptVersion'] == manifest['geval_prompt_version'] == 'geval_audio_plausibility_v6'
    for direction, system_ids in (('a2s', {'ground_truth','ours','flat','cascade'}), ('s2a', {'ground_truth','ours','cascade'})):
        block = geval['summary'][direction]
        assert 'testCount' not in block
        assert {system['id'] for system in block['systems']} == system_ids
        for system in block['systems']:
            assert 'count' not in system
            assert 1 <= system['mean'] <= 5
            assert set(system['criteria']) == set(geval['criterionOrder'][direction])
            for criterion in system['criteria'].values():
                assert 'count' not in criterion
                assert 1 <= criterion['mean'] <= 5
        mismatch = geval['matchedMismatch'][direction]
        assert 'count' not in mismatch
        for variant in ('matched', 'mismatched'):
            assert set(mismatch['scores'][variant]) == set(geval['criterionOrder'][direction])
            assert all(1 <= value <= 5 for value in mismatch['scores'][variant].values())
        assert mismatch['scores']['matched']['scenario_plausibility'] > mismatch['scores']['mismatched']['scenario_plausibility']
    assert all('counts' not in system and 'attempted' not in system for block in data['benchmarks'].values() for system in block['systems'])
    recordings, prompts, judgments, rates = set(), 0, 0, Counter()
    expected_systems = {'a2s': {'ours', 'flat', 'cascade'}, 's2a': {'ours', 'cascade'}}
    for sample in samples:
        original = originals[sample['sample_id']]
        for field in ('audio_caption', 'speech_text', 'speaker_gender', 'speaker_emotion', 'sample_index'):
            assert sample[field] == original[field], (sample['sample_id'], field)
        assert sample['reasoning'].strip() == rationales['a2s'][sample['sample_id']] == rationales['s2a'][sample['sample_id']]
        assert len(sample['files']) == 7
        assert 'scene_group' not in sample and 'emotion_group' not in sample
        assert 'counts' not in data['benchmarks']['a2s']['systems'][0]
        for key, relative in sample['files'].items():
            media = (root / relative).resolve()
            assert media.is_relative_to(root) and media.is_file(), media
            assert digest(media) == digest(args.survey / original['files'][key]), media
            rates[check_wav(media)] += 1
            recordings.add(relative)
        for direction, systems in expected_systems.items():
            assert set(sample['judgments'][direction]) == systems
            for system, judgment in sample['judgments'][direction].items():
                assert set(judgment['criteria']) == set(data['geval']['criterionOrder'][direction])
                assert len(judgment['criteria']) == 5
                values = []
                for criterion in judgment['criteria'].values():
                    assert isinstance(criterion['score'], (int, float)) and 1 <= criterion['score'] <= 5
                    assert criterion['feedback'].strip() and criterion['prompt'].strip()
                    assert criterion['promptVersion'] == manifest['geval_prompt_version']
                    assert '###Score Rubric:' in criterion['prompt']
                    prompts += 1
                    values.append(criterion['score'])
                assert abs(judgment['score'] - sum(values) / 5) < 1e-9
                judgments += 1
    assert len(recordings) == manifest['audio_files']
    pages = list(root.glob('*.html'))
    assert len(pages) == 6
    for page in pages:
        parsed = Page()
        parsed.feed(page.read_text())
        assert len(parsed.ids) == len(set(parsed.ids)), page
        for link in parsed.links:
            target = (page.parent / link).resolve()
            assert target.is_relative_to(root) and target.is_file(), (page, link)
    missing = json.loads((root / 'missing-data.json').read_text())
    assert len(missing) == manifest['missing_fields']
    assert all(not row['field'].startswith('judgments.') for row in missing), 'Missing evaluation evidence'
    print(json.dumps({'result':'pass','pages':len(pages),'matched_ids':len(samples),'identical_source_wavs':len(recordings),'sample_rates':dict(rates),'judgments':judgments,'exact_prompts':prompts,'missing_fields':Counter(row['field'] for row in missing),'total_bytes':sum(p.stat().st_size for p in root.rglob('*') if p.is_file())}, indent=2))

if __name__ == '__main__':
    main()
