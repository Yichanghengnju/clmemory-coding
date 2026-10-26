import hashlib
import math
import re
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient
import pytest

from agent_memory.delivery import coding_tail, source_hit
from agent_memory.submission import build_app


class SyntheticEmbedder:
    dimensions = 64
    identity = {"name": "test-hash-only", "dimensions": 64, "formal_use": False}

    def vector(self, text):
        vector = [0.0] * self.dimensions
        for token in re.findall(r'\w+', text.lower()):
            vector[int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % self.dimensions] += 1
        norm = math.sqrt(sum(x*x for x in vector)) or 1
        return [x/norm for x in vector]

    def embed_documents(self, texts):
        return [self.vector(text) for text in texts]

    def embed_query(self, text):
        return self.vector(text)


def app(path, track="textual"):
    return build_app(db_path=path / (track + '.sqlite3'), api_key='synthetic-key',
                     embedder=SyntheticEmbedder(), track=track, manifest_dir=path / track)


HEADERS = {"Authorization": "Token synthetic-key"}


def body(uid='alice', rid='r1', text='Alice keeps 37 coins in her collection.'):
    return dict(user_id=uid, request_id=rid, session_id='s',
                messages=[dict(role='user', content=text)])


@pytest.mark.parametrize('track', ['textual', 'coding'])
def test_real_native_http_persistence_auth_isolation_and_limits(tmp_path, track):
    instance = app(tmp_path, track)
    with TestClient(instance) as client:
        assert client.get('/health').status_code == 200
        assert client.post('/add', json=body()).status_code == 401
        first = client.post('/add', headers=HEADERS, json=body())
        assert first.status_code == 200 and first.json()['success'] is True
        for headers in (HEADERS, {'Authorization':'Bearer synthetic-key'}, {'X-Api-Key':'synthetic-key'}):
            again = client.post('/add', headers=headers, json=body())
            assert again.json() == first.json()
        assert client.post('/add', headers=HEADERS, json=body(text='changed')).status_code == 409
        for k in (1, 16, 32, 100):
            response = client.post('/search', headers=HEADERS,
                                   json=dict(user_id='alice',query='37 coins',top_k=k))
            assert response.status_code == 200
            hits = response.json()['data']
            assert 1 <= len(hits) <= k
            assert '37 coins' in hits[0]['content']
            assert 'timestamp=unknown' in hits[0]['content']
        assert client.post('/search', headers=HEADERS,
                           json=dict(user_id='bob',query='37 coins',top_k=100)).json() == {'data':[]}
        for bad in (0, 101):
            assert client.post('/search', headers=HEADERS,
                               json=dict(user_id='alice',query='37 coins',top_k=bad)).status_code == 422
    with TestClient(app(tmp_path, track)) as client:
        assert client.post('/search',headers=HEADERS,
                           json=dict(user_id='alice',query='37 coins',top_k=100)).json()['data']


def test_two_tracks_do_not_share_sources(tmp_path):
    with TestClient(app(tmp_path, 'textual')) as textual, TestClient(app(tmp_path,'coding')) as coding:
        assert textual.post('/add',headers=HEADERS,json=body()).status_code == 200
        assert coding.post('/search',headers=HEADERS,
                           json=dict(user_id='alice',query='37 coins',top_k=100)).json() == {'data':[]}


def test_concurrent_identical_add_and_head_preservation(tmp_path):
    with TestClient(app(tmp_path)) as client:
        with ThreadPoolExecutor(max_workers=2) as pool:
            statuses = list(pool.map(lambda _: client.post('/add',headers=HEADERS,json=body()).status_code, range(2)))
        assert statuses == [200, 200]
        payload = body(rid='many')
        payload['messages'] = [dict(role='user', content=f'Alice note {i} about coins.') for i in range(40)]
        assert client.post('/add',headers=HEADERS,json=payload).status_code == 200
        search = lambda k: client.post('/search',headers=HEADERS,
                                       json=dict(user_id='alice',query='Alice coins',top_k=k)).json()['data']
        head, all_hits = search(16), search(100)
        assert all_hits[:len(head)] == head
        assert len({row['id'] for row in all_hits}) == len(all_hits) <= 100


def test_c7_preserves_prefix_and_uses_only_explicit_paths():
    rows = [{'id':f's{i:03d}', 'content':f'[repo/f.py:{i+1}-{i+1}] unique {i}',
             'role':'user','timestamp':None} for i in range(80)]
    rows += [{'id':f'x{i:03d}', 'content':f'no explicit path {i}',
              'role':'assistant','timestamp':None} for i in range(20)]
    sources = {row['id']:row for row in rows}
    ranked = [source_hit(row) for row in rows[:60]] + [source_hit(row) for row in rows[80:]]
    shaped = coding_tail(ranked,sources,100)
    assert shaped[:32] == ranked[:32]
    assert len(shaped) <= len(ranked)
    assert len({sid for hit in shaped for sid in hit.source_ids}) == len(shaped)
    assert all(hit in shaped for hit in ranked[60:])
