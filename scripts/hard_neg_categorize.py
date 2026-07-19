"""分类 high-conf FP 候选：
- 'A. 漏标可能高' (gt_count=0 + conf>0.30)：图本身无标签但模型高置信度预测 = 最可疑漏标
- 'B. 边界可疑'  (gt_count=0 + conf 0.10-0.30)：需要看图确认
- 'C. 有 GT 但模型在别处误报' (gt_count>0)：可能是双重标注/模型在 GT 之外看到东西
按图去重，给每张图最高 conf 的候选
"""
import json
from collections import defaultdict, Counter

CAND = json.load(open('/tmp/v11_hard_neg_candidates.json'))

# 按图分组，每张图保留最高 conf 的候选
by_img = defaultdict(list)
for c in CAND:
    by_img[c['img']].append(c)

per_img_top = {}
for img, cs in by_img.items():
    cs.sort(key=lambda x: -x['conf'])
    per_img_top[img] = cs[0]  # 最高 conf 候选代表这张图

img_summary = []
for img, top in per_img_top.items():
    gt = top['gt_count']
    conf = top['conf']
    if gt == 0 and conf >= 0.30:
        cat = 'A_漏标高可疑'
    elif gt == 0 and conf >= 0.10:
        cat = 'B_边界可疑'
    elif gt > 0:
        cat = 'C_有GT但模型别处误报'
    else:
        cat = '其他'
    img_summary.append({'img': img, 'top_conf': conf, 'gt_count': gt, 'category': cat})

img_summary.sort(key=lambda x: -x['top_conf'])

print('=== 分类汇总 ===')
print(Counter(s['category'] for s in img_summary))
print()

print('=== Top-30 A 类 (gt=0, conf>0.30，最可能是漏标) ===')
print(f'{"conf":>6} {"img":<20} {"bbox_中心":<25} {"宽":>5} {"高":>5}')
a_list = [s for s in img_summary if s['category'] == 'A_漏标高可疑']
for s in a_list[:30]:
    top = per_img_top[s['img']]
    cx, cy = top['center']
    bw = top['bbox'][2] - top['bbox'][0]
    bh = top['bbox'][3] - top['bbox'][1]
    print(f'{s["top_conf"]:>6.3f} {s["img"]:<20} ({cx:>5.0f},{cy:>5.0f}){bw:>5.0f} {bh:>5.0f}')
print(f'\nA 类总数: {len(a_list)}')

print('\n=== Top-20 B 类 (gt=0, conf 0.10-0.30，需要看图) ===')
b_list = [s for s in img_summary if s['category'] == 'B_边界可疑']
for s in b_list[:20]:
    top = per_img_top[s['img']]
    cx, cy = top['center']
    print(f'{s["top_conf"]:>6.3f} {s["img"]:<20} ({cx:>5.0f},{cy:>5.0f})')
print(f'B 类总数: {len(b_list)}')

print('\n=== Top-10 C 类 (有 GT 但模型别处误报，可能是双重目标) ===')
c_list = [s for s in img_summary if s['category'] == 'C_有GT但模型别处误报']
for s in c_list[:10]:
    top = per_img_top[s['img']]
    print(f'{s["top_conf"]:>6.3f} {s["img"]:<20} gt={s["gt_count"]}')
print(f'C 类总数: {len(c_list)}')

# 保存
out = {
    'A_漏标高可疑': [s['img'] for s in a_list],
    'B_边界可疑': [s['img'] for s in b_list],
    'C_有GT但模型别处误报': [s['img'] for s in c_list],
}
json.dump(out, open('/tmp/v11_hard_neg_categories.json', 'w'), indent=1)
print(f'\nSaved分类 to /tmp/v11_hard_neg_categories.json')