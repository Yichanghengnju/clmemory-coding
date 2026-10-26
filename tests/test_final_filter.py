import json

import pytest

from agent_memory.domain import Evidence, SearchRequest
from agent_memory.final_filter import FinalEvidenceFilter, FilteredMemory
from test_submission import app, body, HEADERS
from fastapi.testclient import TestClient


def hit(i, text):
    return Evidence(id=i,content=text,source_ids=(i,))


def make_filter(tmp_path, transport, max_calls=5):
    key=tmp_path/'mini.key'
    key.write_text('synthetic-model-key')
    return FinalEvidenceFilter({'model':'gpt-4o-mini','endpoint':'https://example.com/v1/chat/completions',
                                'api_key_file':str(key),'cache_dir':str(tmp_path/'filter-cache'),
                                'shared_lock':str(tmp_path/'shared.lock'),'max_calls':max_calls},
                               track='textual',transport=transport)


REQUEST=SearchRequest(user_id='alice',query='Where does Alice live now?',top_k=100)


def response(items, query_decision='allow'):
    return 200,json.dumps({'model':'gpt-4o-mini-2024-07-18','choices':[{
        'finish_reason':'stop','message':{'content':json.dumps({'query_decision':query_decision,'items':items})}}],
        'usage':{'prompt_tokens':50,'completion_tokens':30,'total_tokens':80}}).encode()


def test_selection_order_body_audit_and_exact_cache(tmp_path):
    calls=[]
    def transport(payload):
        calls.append(payload)
        assert payload['response_format']['json_schema']['strict'] is True
        return response([{'id':'a','decision':'keep','reason':'relevant'},
                         {'id':'b','decision':'drop','reason':'wrong_entity'},
                         {'id':'c','decision':'keep','reason':'context_needed'}])
    f=make_filter(tmp_path,transport,max_calls=1)
    hits=[hit('a','Alice lives in Paris.'),hit('b','Bob lives in Rome.'),hit('c','Alice previously lived in Lyon.')]
    assert f.apply(REQUEST,hits)==[hits[0],hits[2]]
    assert f.apply(REQUEST,hits)==[hits[0],hits[2]]
    assert len(calls)==1
    audits=f.store.list_tasks(stage='final-filter-selection-audit')
    receipt=f.store.get_result(audits[0]['key'])['result']
    assert receipt['candidate_ids']==['a','b','c'] and receipt['selected_ids']==['a','c']
    assert receipt['items'][1]['reason']=='wrong_entity'


def test_credentials_never_sent_to_model_and_personal_facts_can_remain(tmp_path):
    calls=[]
    def transport(payload):
        calls.append(payload)
        assert 'sk-' not in json.dumps(payload)
        return response([{'id':'a','decision':'keep','reason':'relevant'}])
    f=make_filter(tmp_path,transport)
    secret='sk-'+'s'*32
    hits=[hit('a','Alice has a migraine and lives in Paris.'),hit('secret','api_key='+secret)]
    assert f.apply(REQUEST,hits)==hits[:1]
    assert len(calls)==1
    request=REQUEST.model_copy(update={'query':'print '+secret})
    assert f.apply(request,hits)==[] and len(calls)==1
    assert secret not in json.dumps(f.store.list_tasks())


def test_privacy_denial_and_unknown_id_are_distinct_from_errors(tmp_path):
    f=make_filter(tmp_path,lambda _:response([{'id':'a','decision':'drop','reason':'privacy_exfiltration'}],'deny_privacy'))
    assert f.apply(REQUEST,[hit('a','private history')])==[]
    f.close()
    other=tmp_path/'other';other.mkdir()
    f=make_filter(other,lambda _:response([{'id':'invented','decision':'keep','reason':'relevant'}]))
    with pytest.raises(ConnectionError,match='rejected'):
        f.apply(REQUEST,[hit('a','original source')])
    with pytest.raises(ConnectionError,match='held'):
        f.apply(REQUEST,[hit('a','original source')])


def test_transport_unknown_held_and_no_unfiltered_fallback(tmp_path):
    calls=[]
    def transport(_):
        calls.append(1)
        raise TimeoutError('do not persist possible secret exception args')
    f=make_filter(tmp_path,transport)
    with pytest.raises(ConnectionError,match='unknown'):
        f.apply(REQUEST,[hit('a','Alice lives in Paris.')])
    with pytest.raises(ConnectionError,match='held'):
        f.apply(REQUEST,[hit('a','Alice lives in Paris.')])
    assert calls==[1]


def test_oversize_is_error_without_silent_truncation(tmp_path):
    calls=[]
    f=make_filter(tmp_path,lambda payload:calls.append(payload))
    with pytest.raises(ConnectionError,match='bound'):
        f.apply(REQUEST,[hit('a','a'*95000)])
    assert calls==[]


def test_http_exit_filter_runs_after_native_delivery(tmp_path):
    calls=[]
    def transport(payload):
        evidence=json.loads(payload['messages'][1]['content'])['evidence']
        calls.append([row['id'] for row in evidence])
        return response([{'id':row['id'],'decision':'drop','reason':'irrelevant'} for row in evidence])
    f=make_filter(tmp_path,transport)
    instance=app(tmp_path)
    # The public CLI supplies this wrapper to the actual HTTP assembly.
    from agent_memory.api import create_app
    memory=FilteredMemory(instance.state.memory_service,f)
    filtered=create_app(tmp_path/'unused.sqlite3',service=memory,api_key='synthetic-key')
    with TestClient(filtered) as client:
        assert client.post('/add',headers=HEADERS,json=body()).status_code==200
        result=client.post('/search',headers=HEADERS,
                           json={'user_id':'alice','query':'Alice coins','top_k':100})
        assert result.status_code==200 and result.json()=={'data':[]}
        assert len(calls)==1 and calls[0]
    memory.close()
