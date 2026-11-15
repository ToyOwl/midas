"""Uploads, genuine embedding contract and opt-in reproducible portfolio inputs."""
import base64
import copy
import io
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from portagent.config import load_config
from portagent.materials import analysis, portfolio, storage
from portagent.portfolio.hrp import hrp_weights


@pytest.fixture
def config():
    return load_config(Path(__file__).resolve().parents[1] / 'configs/default.yaml')


def payload():
    stream = io.BytesIO()
    Image.new('RGB', (64, 48), '#808080').save(stream, format='PNG')
    return dict(name='../table.png', content=base64.b64encode(stream.getvalue()).decode(),
                assets=['US:AAPL'], kind='table', source_date='2026-11-01', note='Строка отчётности')


def answer(asset='US:AAPL', relevant=True):
    return dict(content_type='table', relevant=relevant, summary='В таблице снизилась прибыль.',
                visible_text='AAPL Profit', signals=[dict(asset=asset, entity_quote='AAPL', reason='Прибыль снизилась.', risk=.8, confidence=.9)] if relevant else [], limitations=[])


def embedding(_):
    return dict(model='test-model', revision='v1', dimensions=8, vector=[1.]+[0.]*7)


def prepared(root, config, *, relevant=True):
    record = storage.upload(root, payload(), config)
    result = analysis.analyze(root, record['id'], config, chat=lambda *a, **kw: (json.dumps(answer(relevant=relevant)), {}), embed=embedding)
    return {**record, 'analysis': result}


def test_upload_immutable_deduplicated_and_bounded(tmp_path, config):
    first = storage.upload(tmp_path, payload(), config)
    assert storage.upload(tmp_path, payload(), config) == first
    assert (tmp_path / 'images' / (first['id'] + '.png')).is_file()
    assert not (tmp_path.parent / 'table.png').exists()
    assert len(list((tmp_path / 'records').glob('*.json'))) == 1
    with pytest.raises(ValueError, match='перезаписать'):
        storage.immutable(tmp_path / 'originals' / (first['original_sha'] + '.bin'), b'changed')
    for changes in ({'assets':['OUTSIDE']}, {'content':base64.b64encode(b'<svg/>').decode()},
                    {'kind':'execute'}, {'source_date':'2099-01-01'}, {'content':'invalid!!!!'}):
        with pytest.raises(ValueError):
            storage.upload(tmp_path, {**payload(), **changes}, config)


def test_analysis_cached_and_no_bad_tickers_or_numbers(tmp_path, config):
    record = prepared(tmp_path, config)
    cached = analysis.analyze(tmp_path, record['id'], config, chat=lambda *a, **kw: pytest.fail('unexpected model call'))
    assert cached['cache_hit'] and cached['embedding']['dimensions'] == 8
    assert storage.list_materials(tmp_path)[0]['analysis']['embedding_info']['dimensions'] == 8
    assert 'vector' not in storage.list_materials(tmp_path)[0]['analysis']['embedding_info']
    for bad in ('US:UNKNOWN',):
        with pytest.raises(ValueError):
            analysis.validate(answer(bad), ['US:AAPL'])
    for value in (float('nan'), -1, 2, True):
        bad = answer(); bad['signals'][0]['risk'] = value
        with pytest.raises(ValueError):
            analysis.validate(bad, ['US:AAPL'])


def test_failed_analysis_retry_reuses_embedding(tmp_path, config):
    record = storage.upload(tmp_path, payload(), config)
    with pytest.raises(ValueError):
        analysis.analyze(tmp_path, record['id'], config, chat=lambda *a, **kw: ('{}', {}), embed=embedding)
    assert not (tmp_path / 'analyses' / (record['id'] + '.json')).exists()
    done = analysis.analyze(tmp_path, record['id'], config,
        chat=lambda *a, **kw: (json.dumps(answer()), {}), embed=lambda *_: pytest.fail('recomputed vector'))
    assert done['relevant']


def price_inputs():
    assets=['US:AAPL','US:MSFT','US:NVDA','US:JPM']
    rng=np.random.default_rng(18)
    returns=pd.DataFrame(rng.normal(0,.01,(200,4)),columns=assets)
    values=np.full((4,4),.7)
    np.fill_diagonal(values,0)
    D=pd.DataFrame(values,index=assets,columns=assets)
    return returns,D


def test_irrelevant_images_do_not_change_portfolio(tmp_path, config):
    record=prepared(tmp_path,config,relevant=False)
    ret,D=price_inputs()
    changed,distance,meta=portfolio.transform(ret,D,[record],.15)
    pd.testing.assert_frame_equal(ret,changed)
    pd.testing.assert_frame_equal(D,distance)
    assert meta['used_materials']==[]


def test_real_vectors_and_grounded_risk_affect_hrp_deterministically(tmp_path,config):
    one=prepared(tmp_path,config)
    two=copy.deepcopy(one); two.update(id='b'*64,original_sha='c'*64)
    two['analysis']['signals'][0]['asset']='US:MSFT'
    two['analysis']['embedding']['vector']=[0.,1.]+[0.]*6
    ret,D=price_inputs()
    changed,distance,meta=portfolio.transform(ret,D,[one,two],.3)
    assert changed['US:AAPL'].std()>ret['US:AAPL'].std()
    assert distance.loc['US:AAPL','US:MSFT'] != D.loc['US:AAPL','US:MSFT']
    assert distance.loc['US:NVDA','US:JPM'] == D.loc['US:NVDA','US:JPM']
    base=hrp_weights(ret,D,max_weight=.5,cash_share=.1)['weights']
    result=hrp_weights(changed,distance,max_weight=.5,cash_share=.1)['weights']
    assert not np.allclose(base.sort_index(),result.sort_index())
    assert abs(result.sum()-.9)<1e-12 and result.max() <= .45+1e-12
    duplicate=portfolio.transform(ret,D,[one,two,one],.3)
    pd.testing.assert_frame_equal(changed,duplicate[0])
    pd.testing.assert_frame_equal(distance,duplicate[1])
    swapped=portfolio.transform(ret,D,[two,one],.3)
    pd.testing.assert_frame_equal(changed,swapped[0])
    assert meta==swapped[2]


def test_activation_is_future_only_and_daily_inputs_frozen(tmp_path,config,monkeypatch):
    monkeypatch.setattr(portfolio,'today',lambda:date(2026,11,14))
    record=prepared(tmp_path,config)
    scenario={'date':'2026-11-14','records':[record],'details':{'influence':.15,'used_materials':[record['id']]}}
    sid='d'*64
    storage.immutable(tmp_path/'scenarios'/(sid+'.json'),storage.packed(scenario))
    assert portfolio.daily_inputs(tmp_path,'2026-11-14')['records']==[]
    plan=portfolio.activate(tmp_path,sid,7)
    assert plan['effective_from']=='2026-11-15' and plan['expires_on']=='2026-11-21'
    assert portfolio.daily_inputs(tmp_path,'2026-11-14')['records']==[]
    fixed=portfolio.daily_inputs(tmp_path,'2026-11-15')
    assert fixed['records'][0]['id']==record['id']
    portfolio.deactivate(tmp_path)
    assert portfolio.daily_inputs(tmp_path,'2026-11-15')==fixed
    assert portfolio.daily_inputs(tmp_path,'2026-11-16')['records']==[]
    portfolio.activate(tmp_path,sid,7)
    assert portfolio.daily_inputs(tmp_path,'2026-11-22')['records']==[]


def test_mixed_encoder_versions_rejected(tmp_path,config):
    one=prepared(tmp_path,config)
    two=copy.deepcopy(one); two['analysis']['embedding']['revision']='different'
    ret,D=price_inputs()
    with pytest.raises(ValueError,match='разных моделей'):
        portfolio.transform(ret,D,[one,two],.15)


def test_materials_http_security_and_read_access(tmp_path, config):
    from test_web import request
    from portagent.web.server import Application
    cfg=tmp_path/'configs/default.yaml';cfg.parent.mkdir();cfg.write_text(Path('configs/default.yaml').read_text(encoding='utf-8'),encoding='utf-8')
    app=Application(cfg);app.sessions['test']=('csrf-token',9999999999)
    rejected=request(app,'/api/materials/upload',method='POST',data=payload(),headers={'X-CSRF-Token':'wrong'})
    assert rejected['status']==403
    good=request(app,'/api/materials/upload',method='POST',data=payload())
    assert good['status']==201
    image=request(app,'/api/materials/'+good['value']['id']+'/image')
    assert image['mime']=='image/png' and image['value'].startswith(b'\x89PNG')
    assert request(app,'/api/materials')['value']['items'][0]['id']==good['value']['id']
    assert request(app,'/api/materials/upload',method='POST',data=payload(),headers={'Origin':'https://evil.example'})['status']==403
    assert request(app,'/api/materials/activate',method='POST',data={'scenario':'../../outside'})['status']==400


@pytest.mark.parametrize('phrase',['Revenue fell.','Выручка снизилась годом前年.'])
def test_translation_only_changes_prose(tmp_path,config,phrase):
    record=storage.upload(tmp_path,payload(),config)
    response=answer();response['summary']=phrase;response['signals'][0]['reason']='Прибыль снизилась.'
    replies=iter([response,{'texts':['Выручка снизилась.','Прибыль снизилась.']}])
    result=analysis.analyze(tmp_path,record['id'],config,chat=lambda *a,**kw:(json.dumps(next(replies)),{}),embed=embedding)
    assert result['summary']=='Выручка снизилась.'
    assert result['signals'][0]['risk']==.8 and result['signals'][0]['confidence']==.9


def test_unrelated_table_cannot_get_signal_from_ticker_selection(config):
    wrong=answer();wrong['visible_text']='TabM NGBoost';wrong['signals'][0]['entity_quote']='TabM'
    result=analysis.bind_signals(wrong,{'kind':'auto','note':''},config)
    assert result['signals']==[] and not result['relevant']
    invented=answer();invented['visible_text']='Model comparison'
    assert not analysis.bind_signals(invented,{'kind':'auto','note':''},config)['relevant']
    chart=answer();chart['visible_text']='Price';chart['signals'][0]['entity_quote']='';chart['content_type']='chart'
    assert analysis.bind_signals(chart,{'kind':'chart','note':''},config)['signals'][0]['association']=='user_chart_assignment'


def test_preview_replaces_existing_materials_without_stacking(tmp_path, config, monkeypatch):
    from portagent.report import diagnostics
    from portagent.features.prices import log_returns
    root=tmp_path/'data/materials';record=prepared(root,config)
    ret,D=price_inputs()
    panel=100*np.exp(ret.cumsum());panel.index=pd.date_range('2025-01-01',periods=len(panel)).strftime('%Y-%m-%d')
    returns=log_returns(panel)
    monkeypatch.setattr(diagnostics,'archived_prices',lambda paths,day:{a:pd.DataFrame({'close':panel[a]}) for a in panel})
    changed,dist,meta=portfolio.transform(returns,D,[record],.15)
    settings=dict(linkage_method='single',max_weight=.5,cash_share=.1)
    weights=hrp_weights(changed,dist,**settings)['weights'].to_dict()
    report=dict(date='2026-11-14',weights=weights,portfolio={'linkage_method':'single','max_weight':.5,'cash_share':.1},
                fusion={'views_used':['price'],'method':'linear','w_price':.7})
    paths={'data':tmp_path/'data','reports':tmp_path/'reports'}
    storage.immutable(paths['reports']/report['date']/'portfolio.json',storage.packed(report))
    storage.immutable(paths['data']/'forecasts'/('2026-11-14.json'),storage.packed({'views':{'price':D.to_dict()},'materials':{'applied':True,'plan':'original'}}))
    storage.immutable(root/'daily/2026-11-14.json',storage.packed({'records':[record],'influence':.15,'plan':'original'}))
    before=(paths['reports']/report['date']/'portfolio.json').read_bytes()
    result=portfolio.preview(paths,report,config,[record['id']],.15)
    assert result['turnover']<1e-12
    again=portfolio.preview(paths,report,config,[record['id']],.15)
    assert result==again
    assert (paths['reports']/report['date']/'portfolio.json').read_bytes()==before


def test_visual_evidence_distinguishes_upload_from_applied_plan(tmp_path,config):
    from portagent.llm.critic import build_evidence
    record=prepared(tmp_path,config)
    snapshot={'weights':{'US:AAPL':.2},'uploaded_materials':[record]}
    first=build_evidence(snapshot)
    assert any(r['id'].startswith('EV-IMAGE-') for r in first)
    assert not any(r['id']=='EV-IMAGES-APPLIED' for r in first)
    snapshot['material_effects']={'applied':True,'influence':.15,'reasons':{'US:AAPL':[{'text':'Прибыль снизилась.'}]}}
    assert any(r['id']=='EV-IMAGES-APPLIED' for r in build_evidence(snapshot))


def test_daily_consumes_approved_materials_and_keeps_limits(tmp_path,config,monkeypatch):
    from conftest import synthetic_frames
    from portagent.pipeline import daily
    frames=synthetic_frames()
    monkeypatch.setattr(daily,'_load_fixture',lambda *a:(frames,[],[],[]))
    monkeypatch.setattr(daily,'probe',lambda *a:{'state':'ready'})
    monkeypatch.setattr(daily,'make_chat',lambda *a,**kw:lambda *a,**kw:('{}',{}))
    monkeypatch.setattr(portfolio,'today',lambda:date(2026,11,14))
    cfgpath=tmp_path/'configs/default.yaml';cfgpath.parent.mkdir();cfgpath.write_text('unused')
    baseline_cfg=copy.deepcopy(config);baseline_cfg['paths']={'data_dir':'base-data','reports_dir':'base-reports'}
    baseline=daily.run_daily(baseline_cfg,cfgpath,date='2026-11-15',source='fixture',use_llm=False)['report']
    record=prepared(tmp_path/'data/materials',config)
    sid='a'*64
    storage.immutable(tmp_path/'data/materials/scenarios'/(sid+'.json'),storage.packed({
        'date':'2026-11-14','records':[record],'details':{'influence':.3,'used_materials':[record['id']]}}))
    portfolio.activate(tmp_path/'data/materials',sid,7)
    result=daily.run_daily(config,cfgpath,date='2026-11-15',source='fixture',use_llm=True)['report']
    assert result['materials']['applied']
    assert result['weights'] != baseline['weights']
    assert abs(sum(result['weights'].values())+.1-1)<1e-10
    assert max(result['weights'].values())<=.25*.9+1e-10
    again=daily.run_daily(config,cfgpath,date='2026-11-15',source='fixture',use_llm=True)['report']
    assert result['weights']==again['weights']
    frozen=json.loads((tmp_path/'data/forecasts/2026-11-15.json').read_text(encoding='utf-8'))
    assert frozen['materials']['plan']==result['materials']['plan']
