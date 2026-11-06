"""
06_patient_cluster_bootstrap.py
===============================
Step 6 of the analysis pipeline. Patient-clustered bootstrap for every interval
and p value reported in the manuscript.

Manuscript
----------
Methods 2.5 (uncertainty); Table 2; Results 3.2 to 3.5; Supporting Information
Tables S4 to S6 and S8 to S11.

Design
------
* The fitted models, their test-set predictions and the operating thresholds
  are fixed. The intervals therefore describe uncertainty in evaluating the
  developed models on this test sample, not in developing them.
* The 253 test patients are resampled with replacement (10 000 resamples, seed
  42), and every episode of a sampled patient is included each time that patient
  is drawn. The estimand stays episode-level performance. The same resamples are
  used for every model, so comparisons are paired.
* Standalone estimates (AUROC, AUPRC, Brier score, ICI, sensitivity,
  specificity, predictive values, observed proportions): percentile 95% CIs.
* Differences between models: 95% CI = estimate +/- 1.96 bootstrap standard
  errors, with the matching two-sided Wald p value.
* An equal-size episode bootstrap is run alongside as a control. Its first 1000
  resamples must reproduce the episode-bootstrap intervals of
  05_heldout_statistics.py exactly, and the point estimates must equal those of
  steps 3 and 5. These assertions guard against drift between steps.
* The follow-up analysis uses all 1300 analysed episodes, resampling patients.
* Class-weighting sensitivity: unweighted XGBoost is refitted on the
  development set with the tuned hyperparameters, as in
  05_heldout_statistics.py, and must reproduce its AUROC.

Inputs
------
  data_strict/processed_episodes.csv                           (01)
  results_strict/test_predictions.csv, results_strict_nomicro/test_predictions.csv,
  results_strict/models/XGB_pipeline.joblib                    (02)
  results_strict_benchmarks/test_predictions_all_models.csv,
  results_strict_benchmarks/A_test_benchmarks.csv              (03)
  results_strict_sensitivity/test_predictions.csv,
  results_strict_sensitivity/sensitivity_missingness.csv       (04)
  results_strict_benchmarks/heldout/*.csv and summary.json     (05)

Outputs (results_strict_benchmarks/patient_cluster_bootstrap/, aggregate only)
------------------------------------------------------------------------------
  model_performance.csv, AUROC_comparison.csv, probability_comparison.csv,
  subgroup_performance.csv (sex and age), subgroup_heterogeneity.csv,
  age_joint_tests.csv, missingness_performance.csv,
  missingness_comparison.csv, class_weighting.csv, followup_rate.csv,
  followup_comparison.csv, all_estimates.csv,
  run_metadata.json (input file hashes, package versions, checks)
  bootstrap_metrics_*.npz      cached bootstrap distributions (aggregate)

Usage
-----
  python 06_patient_cluster_bootstrap.py --n-boot 10000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
import xgboost as xgb
from scipy.interpolate import interp1d
from scipy.stats import norm, chi2
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupShuffleSplit
from statsmodels.nonparametric.smoothers_lowess import lowess
from threadpoolctl import threadpool_limits, threadpool_info

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results_strict_benchmarks'
OLD = RES / 'heldout'
MODELS = ['XGB_micro', 'XGB_nomicro', 'LR_micro', 'LR_nomicro', 'VA_only']
Z95 = norm.ppf(.975)
SOURCES: set[Path] = set()


def read(path, **kwargs):
    path = ROOT / path
    SOURCES.add(path)
    return pd.read_csv(path, **kwargs)


def keyed(df):
    df = df.copy()
    assert df.rveeh_ur.notna().all()
    df['rveeh_ur'] = df.rveeh_ur.astype(str)
    df['admission_date'] = pd.to_datetime(df.admission_date)
    assert not df.duplicated(['rveeh_ur', 'admission_date']).any()
    return df


def load_data():
    canonical = keyed(read('results_strict/test_predictions.csv', dtype={'rveeh_ur': str}))
    no = keyed(read('results_strict_nomicro/test_predictions.csv', dtype={'rveeh_ur': str}))
    preds = read('results_strict_benchmarks/test_predictions_all_models.csv')
    full_native = read('data_strict/processed_episodes.csv').dropna(subset=['poor_outcome']).reset_index(drop=True)
    split_groups = full_native.rveeh_ur.to_numpy()
    full = keyed(full_native)
    split = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=61)
    tri, tei = next(split.split(full, full.poor_outcome, split_groups))
    ids = ['rveeh_ur', 'admission_date']
    assert canonical[ids].equals(full.iloc[tei][ids].reset_index(drop=True))
    assert canonical[ids].equals(no[ids])
    assert set(full.iloc[tri].rveeh_ur).isdisjoint(set(canonical.rveeh_ur))
    assert len(preds) == len(canonical) == len(tei)   # 262 in the reported analysis
    assert np.array_equal(preds.y_true, canonical.y_true)
    assert np.array_equal(preds.y_true, no.y_true)
    assert np.allclose(preds.pres_logmar, canonical.pres_logmar, equal_nan=True)
    for m in ['XGB']:
        assert np.allclose(preds[f'prob_{m}_micro'], canonical[f'prob_{m}'], atol=1e-7, rtol=0)
        assert np.allclose(preds[f'prob_{m}_nomicro'], no[f'prob_{m}'], atol=1e-7, rtol=0)
    test = full.iloc[tei].reset_index(drop=True)
    counts = test.groupby('rveeh_ur', sort=False).size()
    meta = dict(test_episodes=len(test), test_patients=len(counts),
                repeat_test_patients=int((counts > 1).sum()),
                episodes_in_repeat_patients=int(counts[counts > 1].sum()),
                largest_patient_cluster=int(counts.max()),
                test_poor=int(preds.y_true.sum()),
                missing_test_va=int(preds.pres_logmar.isna().sum()),
                missing_test_admission_dates=int(test.admission_date.isna().sum()),
                development_test_patient_overlap=0)
    return full, full.iloc[tri], test, preds, meta


def make_samples(groups, n_boot, seed, clustered):
    n = len(groups)
    codes, unique = pd.factorize(groups, sort=False)
    members = [np.flatnonzero(codes == j) for j in range(len(unique))]
    rng = np.random.default_rng(seed)
    weights = np.ones((n_boot + 1, n), dtype=np.float64)
    indices = [np.arange(n)]
    for b in range(1, n_boot + 1):
        if clustered:
            draw = rng.integers(0, len(unique), size=len(unique))
            idx = np.concatenate([members[j] for j in draw])
        else:
            idx = rng.integers(0, n, size=n)
        weights[b] = np.bincount(idx, minlength=n)
        indices.append(idx)
    if clustered:
        for ix in members:
            assert np.all(weights[:, ix] == weights[:, ix[:1]])
        assert np.all(weights[:, [ix[0] for ix in members]].sum(axis=1) == len(unique))
    return weights, indices


def ratio(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.divide(a, b, out=np.full(np.broadcast_shapes(a.shape, b.shape), np.nan), where=b != 0)


def rank_metrics(y, p, weights):
    """Weighted empirical ROC area and average precision, with exact tie groups."""
    order = np.argsort(p, kind='stable')
    starts = np.r_[0, np.flatnonzero(np.diff(p[order]) != 0) + 1]
    wp = np.add.reduceat(weights[:, order] * y[order], starts, axis=1)
    wn = np.add.reduceat(weights[:, order] * (1 - y[order]), starts, axis=1)
    npos, nneg = wp.sum(axis=1), wn.sum(axis=1)
    auc = ratio((wp * (np.cumsum(wn, axis=1) - .5 * wn)).sum(axis=1), npos * nneg)
    pos, neg = np.cumsum(wp[:, ::-1], axis=1), np.cumsum(wn[:, ::-1], axis=1)
    precision = np.divide(pos, pos + neg, out=np.zeros_like(pos), where=(pos + neg) > 0)
    ap = ratio((wp[:, ::-1] * precision).sum(axis=1), npos)
    return auc, ap


def ici(y, p):
    # Same definition, span, interpolation and iteration settings as compute_ici() in
    # evaluation_metrics.py.
    smooth = lowess(y.astype(float), p, frac=.75, it=0, return_sorted=True)
    fn = interp1d(smooth[:, 0], smooth[:, 1], bounds_error=False,
                  fill_value=(smooth[0, 1], smooth[-1, 1]))
    return float(np.mean(abs(fn(p) - p)))


def worker_init():
    global _WORKER_LIMITS
    _WORKER_LIMITS = threadpool_limits(limits=1)


def ici_chunk(args):
    y, p, indices = args
    return np.array([ici(y[idx], p[idx]) for idx in indices])


def operating(y, positive, w):
    tp, fn = w @ (y * positive), w @ (y * ~positive)
    tn, fp = w @ ((1 - y) * ~positive), w @ ((1 - y) * positive)
    return dict(sensitivity=ratio(tp, tp + fn), specificity=ratio(tn, tn + fp),
                ppv=ratio(tp, tp + fp), npv=ratio(tn, tn + fn),
                accuracy=ratio(tp + tn, tp + tn + fp + fn))


def intervals(v, null=None):
    point, draws = float(v[0]), np.asarray(v[1:])
    draws = draws[np.isfinite(draws)]
    if len(draws) < 2:
        return dict(point=point, valid_boot=len(draws))
    se = np.std(draws, ddof=1)
    result = dict(point=point, se=se, lo=point-Z95*se, hi=point+Z95*se,
                  percentile_lo=np.percentile(draws, 2.5), percentile_hi=np.percentile(draws, 97.5),
                  valid_boot=len(draws), bias=float(draws.mean()-point))
    if null is not None:
        result['p'] = 2 * norm.sf(abs((point-null)/se)) if se else (1.0 if point == null else 0.0)
    return result


def self_check():
    # Includes mixed-outcome repeat patients, tied scores and variable cluster sizes.
    groups = np.array(['a', 'a', 'b', 'c', 'c', 'c', 'd'])
    y = np.array([0, 1, 1, 0, 1, 0, 1])
    p = np.array([.3, .3, .7, .5, .6, .6, .4])
    w, ix = make_samples(groups, 30, 42, True)
    auc, ap = rank_metrics(y, p, w)
    for b, idx in enumerate(ix):
        if len(np.unique(y[idx])) == 2:
            assert np.isclose(auc[b], roc_auc_score(y[idx], p[idx]), atol=1e-12)
            assert np.isclose(ap[b], average_precision_score(y[idx], p[idx]), atol=1e-12)
    # With one episode per patient, both procedures must coincide exactly.
    ew, ei = make_samples(np.arange(7), 30, 42, False)
    cw, ci = make_samples(np.arange(7), 30, 42, True)
    assert np.array_equal(ew, cw) and all(np.array_equal(a, b) for a, b in zip(ei, ci))


def main(pool, original_threadpools):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-boot', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=Path, default=RES/'patient_cluster_bootstrap')
    args = parser.parse_args()
    if args.n_boot < 1000:
        raise ValueError('Use at least 1000 replicates.')
    out = args.out.resolve()
    if ROOT not in out.parents or out == OLD.resolve():
        raise ValueError('Use a distinct output directory in this workspace.')
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    self_check()
    full, development, test, preds, meta = load_data()
    y = preds.y_true.to_numpy(int)
    P = {m: preds[f'prob_{m}'].to_numpy(float) for m in MODELS}
    samples = {name: make_samples(test.rveeh_ur, args.n_boot, args.seed, clustered)
               for name, clustered in [('episode', False), ('patient', True)]}
    old_cal = read('results_strict_benchmarks/heldout/calibration_by_model.csv').set_index('model')
    bench = read('results_strict_benchmarks/A_test_benchmarks.csv').set_index('model')
    rows, store = [], {}

    def add(analysis, label, metric, e, c, null=None, old=None, **extra):
        assert np.isclose(e[0], c[0], rtol=1e-9, atol=1e-10, equal_nan=True)
        row = dict(analysis=analysis, label=label, metric=metric, point=float(c[0]), null=null, **extra)
        for name, v in [('episode', e), ('patient', c)]:
            row.update({f'{name}_{k}': val for k, val in intervals(v, null).items() if k != 'point'})
        if old:
            row.update({f'original_{k}': v for k, v in old.items()})
        if 'patient_se' in row and row['episode_se'] > 0:
            row['se_ratio_patient_episode'] = row['patient_se'] / row['episode_se']
        rows.append(row)

    def calculate_model(m, p, ici_needed=True):
        cache_path = out/f'bootstrap_metrics_{m}.npz'
        # v2: no calibration intercept or slope (2 November 2026)
        cache_key = hashlib.sha256(p.tobytes()+y.tobytes()+
                     test.rveeh_ur.to_csv(index=False).encode()+
                     f'{args.n_boot},{args.seed},v2'.encode()).hexdigest()
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as cached:
                if str(cached['cache_key']) == cache_key:
                    for method in samples:
                        store[method, m] = {key.split('__', 1)[1]: cached[key] for key in cached.files if key.startswith(method+'__')}
                    print(f'Verified-input aggregate cache: {m}', flush=True)
                    return
        for method, (w, indices) in samples.items():
            a, ap = rank_metrics(y, p, w)
            v = dict(AUROC=a, AUPRC=ap, Brier=(w @ ((y-p)**2))/w.sum(axis=1))
            if ici_needed:
                chunks = [(y, p, indices[j:j+250]) for j in range(0, len(indices), 250)]
                v['ICI'] = np.concatenate(list(pool.map(ici_chunk, chunks)))
            if m in bench.index:
                v.update(operating(y, p >= bench.loc[m, 'threshold'], w))
            store[method, m] = v
        np.savez_compressed(cache_path, cache_key=cache_key,
                            **{method+'__'+k: v for method in samples for k, v in store[method, m].items()})
        print(f'Metrics ready: {m}; elapsed {time.monotonic()-start:.0f}s', flush=True)

    print(f'Validated data: {meta}; {args.n_boot} replicates per method', flush=True)
    for m in MODELS:
        calculate_model(m, P[m])
        old = old_cal.loc[m]
        for metric in store['patient', m]:
            old_result = None
            key = {'AUROC': 'auroc', 'Brier': 'brier', 'ICI': 'ici'}.get(metric)
            if key:
                old_result = dict(point=old[key], lo=old[f'{key}_lo'], hi=old[f'{key}_hi'],
                                  ci_method='percentile_episode_bootstrap_1000')
                assert np.isclose(store['patient', m][metric][0], old[key], atol=1e-7)
                if metric == 'AUROC':
                    # Exact original-bootstrap reproduction before increasing B/changing unit.
                    q = np.percentile(store['episode', m]['AUROC'][1:1001], [2.5, 97.5])
                    assert np.allclose(q, [old.auroc_lo, old.auroc_hi], atol=1e-12)
            add('model_performance', m, metric, store['episode', m][metric], store['patient', m][metric],
                old=old_result)

    comparisons = read('results_strict_benchmarks/heldout/auroc_comparisons.csv')
    for r in comparisons.itertuples():
        e = store['episode', r.model_a]['AUROC'] - store['episode', r.model_b]['AUROC']
        c = store['patient', r.model_a]['AUROC'] - store['patient', r.model_b]['AUROC']
        assert np.isclose(c[0], r.diff, atol=1e-12)
        assert np.allclose(np.percentile(e[1:1001], [2.5, 97.5]), [r.lo, r.hi], atol=1e-12)
        add('AUROC_comparison', f'{r.model_a} - {r.model_b}', 'AUROC difference', e, c, 0,
            dict(point=r.diff, lo=r.lo, hi=r.hi, p=r.p, ci_method='percentile_episode_bootstrap_1000', p_method='DeLong'))
    for r in read('results_strict_benchmarks/heldout/accuracy_calibration_comparisons.csv').itertuples():
        add('probability_comparison', f'{r.model_a} - {r.model_b}', r.metric+' difference',
            store['episode', r.model_a][r.metric]-store['episode', r.model_b][r.metric],
            store['patient', r.model_a][r.metric]-store['patient', r.model_b][r.metric], 0,
            dict(point=r.diff, lo=r.lo, hi=r.hi, p=r.p,
                 ci_method='percentile_episode_bootstrap_1000', p_method='episode_bootstrap_SE_Wald'))

    # Discrimination by sex and age (fairness). Subgroup bootstraps draw whole
    # patients from the full test sample and then select eligible episodes. This
    # preserves dependence across subgroup contrasts.
    masks = {'Female': test.gender_female.to_numpy() == 1, 'Male': test.gender_female.to_numpy() == 0,
             'Age <65': test.age_years.to_numpy() < 65,
             'Age 65 to 79': (test.age_years.to_numpy() >= 65) & (test.age_years.to_numpy() < 80),
             'Age >=80': test.age_years.to_numpy() >= 80}
    sub = {}
    for name, mask in masks.items():
        if min(y[mask].sum(), (1-y[mask]).sum()) < 5:
            continue
        for m in ['XGB_micro', 'XGB_nomicro', 'LR_micro', 'VA_only']:
            for method, (w, _) in samples.items():
                sub[method, name, m] = rank_metrics(y[mask], P[m][mask], w[:, mask])[0]
            add('subgroup_performance', f'{name}: {m}', 'AUROC', sub['episode', name, m], sub['patient', name, m],
                n=int(mask.sum()))
    group_pairs = [('Female vs male', 'Female', 'Male')]
    old_het = read('results_strict_benchmarks/heldout/subgroup_heterogeneity.csv').set_index(['comparison', 'model'])
    for label, a, b in group_pairs:
        for m in ['XGB_micro', 'XGB_nomicro']:
            r = old_het.loc[label, m]
            add('subgroup_heterogeneity', f'{label}: {m}', 'AUROC difference',
                sub['episode', a, m]-sub['episode', b, m], sub['patient', a, m]-sub['patient', b, m], 0,
                dict(point=r['diff'], p=r.p, p_method='independent_DeLong_variance_z'))
    joint_rows = []
    for m in ['XGB_micro', 'XGB_nomicro']:
        row = dict(model=m, comparison='Age <65 vs 65-79 vs >=80',
                   original_p=old_het.loc['Age <65 vs 65-79 vs >=80 (heterogeneity)', m]['p'])
        for method in samples:
            v = np.column_stack([sub[method, 'Age <65', m]-sub[method, 'Age >=80', m],
                                 sub[method, 'Age 65 to 79', m]-sub[method, 'Age >=80', m]])
            valid = v[1:][np.isfinite(v[1:]).all(axis=1)]
            cov = np.cov(valid.T, ddof=1)
            stat = float(v[0] @ np.linalg.solve(cov, v[0]))
            row.update({method+'_p': chi2.sf(stat, 2), method+'_chi2': stat, method+'_valid_boot': len(valid)})
        joint_rows.append(row)

    # Existing ablation predictions: no fitting or retuning.
    ablation = read('results_strict_sensitivity/test_predictions.csv')
    assert np.array_equal(ablation.y_true, y)
    old_ab = read('results_strict_sensitivity/sensitivity_missingness.csv').set_index(['model', 'tier'])
    for model in ['XGB']:
        assert np.allclose(ablation[f'prob_{model}_full'], P[model+'_micro'], atol=1e-7, rtol=0)
        for tier in ['drop_immune', 'drop_immune_rapd']:
            p = ablation[f'prob_{model}_{tier}'].to_numpy()
            vals = {method: rank_metrics(y, p, w)[0] for method, (w, _) in samples.items()}
            r = old_ab.loc[model, tier]
            add('missingness_performance', f'{model}: {tier}', 'AUROC', vals['episode'], vals['patient'],
                old=dict(point=r.test_auroc, lo=r.auroc_lower_95, hi=r.auroc_upper_95))
            # Original ablation table uses FULL minus REDUCED, retained here.
            add('missingness_comparison', f'{model}: full - {tier}', 'AUROC difference',
                store['episode', model+'_micro']['AUROC']-vals['episode'],
                store['patient', model+'_micro']['AUROC']-vals['patient'], 0,
                dict(point=r.auroc_diff_vs_full, lo=r.diff_lower_95, hi=r.diff_upper_95, p=r.delong_p, p_method='DeLong'))

    # Recreate only the previously specified unweighted sensitivity model.
    # Canonical models/predictions remain untouched; no search or bootstrap refitting.
    old_weight = read('results_strict_benchmarks/heldout/class_weighting.csv').set_index(['model', 'metric'])
    for model in ['XGB']:
        path = ROOT/f'results_strict/models/{model}_pipeline.joblib'
        SOURCES.add(path)
        fitted = joblib.load(path)
        features = list(fitted.feature_names_in_)
        assert np.allclose(fitted.predict_proba(test[features])[:, 1], P[model+'_micro'], atol=1e-7, rtol=0)
        pipe = clone(fitted)
        # Mirror 05_heldout_statistics.py exactly. Its reconstruction copies only
        # the tuned parameters, letting all other estimator defaults reset.
        hp_keys = ['n_estimators','max_depth','learning_rate','subsample','colsample_bytree',
                   'min_child_weight','gamma','reg_alpha','reg_lambda']
        hp = {k: fitted.named_steps['clf'].get_params()[k] for k in hp_keys}
        estimator = xgb.XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1,
                                      scale_pos_weight=1.0, **hp)
        pipe.set_params(clf=estimator)
        # The original XGBoost fit is numerically sensitive to changing thread
        # settings. Restore the environment used by the original fitting scripts.
        with threadpool_limits(limits=original_threadpools):
            pipe.fit(development[features], development.poor_outcome.astype(int))
            p = pipe.predict_proba(test[features])[:, 1].astype(float)
        # Original XGBoost comparison computed these sensitivity predictions in
        # float32. Use float64 for reproducible numerical arithmetic throughout.
        assert np.isclose(roc_auc_score(y, p), old_weight.loc[model, 'AUROC']['unweighted'], atol=1e-10)
        name = model+'_unweighted'
        calculate_model(name, p)
        for metric in ['AUROC', 'Brier', 'ICI']:
            r = old_weight.loc[model, metric]
            add('class_weighting', model+': unweighted - weighted', metric+' difference',
                store['episode', name][metric]-store['episode', model+'_micro'][metric],
                store['patient', name][metric]-store['patient', model+'_micro'][metric], 0,
                dict(point=r['diff'], lo=r.lo, hi=r.hi, p=r.p,
                     p_method='DeLong' if metric == 'AUROC' else 'episode_bootstrap_SE_Wald'))

    # Descriptive follow-up association uses the full cohort, not just test data.
    # Replace the episode chi-square test by a clustered bootstrap Wald test.
    SOURCES.add(OLD/'summary.json')
    old_summary = json.loads((OLD/'summary.json').read_text())
    yy = full.poor_outcome.to_numpy(int)
    fu = full.days_to_final_visit.to_numpy(float)
    full_vals = {}
    for method, clustered in [('episode', False), ('patient', True)]:
        w, _ = make_samples(full.rveeh_ur, args.n_boot, args.seed, clustered)
        vals = {}
        for label, mask in [('>=180', fu >= 180), ('<180', fu < 180)]:
            vals[label] = ratio(w@(yy*mask), w@mask.astype(float))
        full_vals[method] = vals
    add('followup_comparison', '>=180 days - <180 days', 'poor-outcome rate difference',
        full_vals['episode']['>=180']-full_vals['episode']['<180'], full_vals['patient']['>=180']-full_vals['patient']['<180'], 0,
        dict(p=old_summary['followup']['chi_p'], p_method='chi_square_with_continuity_correction'))
    for label in full_vals['patient']:
        add('followup_rate', label, 'poor-outcome rate', full_vals['episode'][label], full_vals['patient'][label])

    frame = pd.DataFrame(rows)
    frame.to_csv(out/'all_estimates.csv', index=False)
    for analysis, data in frame.groupby('analysis', sort=False):
        data.to_csv(out/(analysis+'.csv'), index=False)
    pd.DataFrame(joint_rows).to_csv(out/'age_joint_tests.csv', index=False)
    # Ensure original point estimates were preserved to supplied precision.
    exact = frame.original_point.notna() & ~frame.analysis.str.startswith('missingness')
    assert np.allclose(frame.loc[exact, 'point'], frame.loc[exact, 'original_point'], atol=1e-6, rtol=0)
    assert frame.patient_valid_boot.min() >= .95*args.n_boot
    assert frame.episode_valid_boot.min() >= .95*args.n_boot
    meta.update(n_boot=args.n_boot, seed=args.seed, elapsed_seconds=time.monotonic()-start,
                scope='Fixed original test predictions and thresholds; full-cohort descriptive follow-up; no primary model refit.',
                ci_for_p_values='point +/- 1.95996398454 * bootstrap SD; two-sided normal Wald p',
                additional_ci='percentile intervals separately labelled; no p paired to these',
                sampling='Unstratified subjects with replacement; retain all episodes each selection; same draws for all model pairs.',
                episode_estimand_retained=True, uncertainty_conditions_on_development=True,
                multiplicity='No adjustment for multiple comparisons.',
                primary_predictions_changed=False, canonical_result_files_changed=False,
                sensitivity_refits='Only the existing XGB unweighted arm recreated with frozen hyperparameters on original development data.',
                sensitivity_refit_thread_settings=original_threadpools,
                checks=['episode 1000-bootstrap AUROC/difference CIs reproduce canonical outputs',
                        'patient/episode/feature/prediction alignment verified',
                        'weighted AUC/AP verified against expanded samples with ties',
                        'one-episode-per-patient sampler identity verified',
                        'whole-cluster multiplicities verified', 'original point estimates preserved'],
                versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                              scipy=scipy.__version__, sklearn=sklearn.__version__, statsmodels=statsmodels.__version__),
                source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCES)},
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                max_invalid_patient_boot=int(args.n_boot-frame.patient_valid_boot.min()),
                min_test_boot_episodes=int(samples['patient'][0][1:].sum(axis=1).min()),
                max_test_boot_episodes=int(samples['patient'][0][1:].sum(axis=1).max()))
    (out/'run_metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(frame[frame.analysis.eq('AUROC_comparison')][['label','point','original_p','episode_p','patient_lo','patient_hi','patient_p','se_ratio_patient_episode']].to_string(index=False), flush=True)
    print(f'Completed: {len(frame)} scalar estimates; {out}', flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    original_threadpools = threadpool_info()
    with threadpool_limits(limits=1), ProcessPoolExecutor(max_workers=4, initializer=worker_init) as pool:
        main(pool, original_threadpools)
