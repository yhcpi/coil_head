"""从 train 预测 JSON 找 high-conf FP 候选（漏标可疑）"""
import json
import math
from pathlib import Path

PRED_JSON = '/tmp/v11_train_predictions.json'
OUT_JSON = '/tmp/v11_hard_neg_candidates.json'
CONF_THR = 0.10
LENIENT_DIST = 30.0

def center(b):
    return ((b[0]+b[2])/2, (b[1]+b[3])/2)

def dist(c1, c2):
    return math.hypot(c1[0]-c2[0], c1[1]-c2[1])

data = json.load(open(PRED_JSON))
candidates = []

for r in data:
    gts = r['gt_boxes']
    for p in r['preds']:
        if p['conf'] < CONF_THR:
            continue
        pc = center([p['x1'], p['y1'], p['x2'], p['y2']])
        matched = False
        for g in gts:
            gc = center(g)
            if dist(pc, gc) <= LENIENT_DIST:
                matched = True
                break
        if not matched:
            candidates.append({
                'img': r['img'],
                'W': r['W'], 'H': r['H'],
                'conf': p['conf'],
                'bbox': [p['x1'], p['y1'], p['x2'], p['y2']],
                'center': [pc[0], pc[1]],
                'gt_count': r['gt_count'],
                'has_close_gt': False,
            })

candidates.sort(key=lambda x: -x['conf'])
print('Total candidates (conf>={}, no GT within {}px): {}'.format(CONF_THR, LENIENT_DIST, len(candidates)))
print('\nTop-20 by conf:')
print('{:>6} {:<15} {:>3} {:>6} {:>6} {:>5} {:>5}'.format('conf', 'img', 'gt', 'cx', 'cy', 'bw', 'bh'))
for c in candidates[:20]:
    cx, cy = c['center']
    bw = c['bbox'][2]-c['bbox'][0]
    bh = c['bbox'][3]-c['bbox'][1]
    print('{:>6.3f} {:<15} {:>3} {:>6.0f} {:>6.0f} {:>5.0f} {:>5.0f}'.format(
        c['conf'], c['img'], c['gt_count'], cx, cy, bw, bh))

bins = [(0.10, 0.20), (0.20, 0.30), (0.30, 0.50), (0.50, 0.70), (0.70, 0.95)]
print('\n按 conf 区间分布:')
for lo, hi in bins:
    n = sum(1 for c in candidates if lo <= c['conf'] < hi)
    print('  [{:.2f}, {:.2f}): {} 张'.format(lo, hi, n))

print('\n按 gt_count 分布:')
gt_zero = sum(1 for c in candidates if c['gt_count']==0)
gt_nonzero = sum(1 for c in candidates if c['gt_count']>0)
print('  gt_count=0 (空标签): {} 个候选'.format(gt_zero))
print('  gt_count>0 (有标签但模型在别处误报): {} 个候选'.format(gt_nonzero))

OUT = Path(OUT_JSON)
OUT.write_text(json.dumps(candidates, indent=1))
print('\nSaved {} candidates to {}'.format(len(candidates), OUT_JSON))
