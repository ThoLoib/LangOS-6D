import json, os

base = '/home/tholoi/thesis/OSCAR/object_database'
desc_base = base + '/descriptions_tessa'

# MI3DOR: check category breakdown
d4 = json.load(open(desc_base + '/MI3DOR/descriptions_attributes.json'))
desc_keys = set(d4.keys())

mi3dor_base = base + '/MI3DOR/model/test'
model_keys = set()
cat_counts = {}
for cat in sorted(os.listdir(mi3dor_base)):
    cat_path = mi3dor_base + '/' + cat
    if os.path.isdir(cat_path):
        files = [os.path.splitext(f)[0] for f in os.listdir(cat_path) if f.endswith('.glb') or f.endswith('.obj')]
        cat_counts[cat] = len(files)
        model_keys.update(files)

print('MI3DOR categories and counts (models vs descriptions):')
for cat, cnt in sorted(cat_counts.items()):
    desc_in_cat = len([k for k in desc_keys if k.startswith(cat)])
    print(f'  {cat}: {cnt} models, {desc_in_cat} in desc')

print()
print('Total models:', len(model_keys))
print('Total desc keys:', len(desc_keys))

# Check naming pattern difference
extra_models = sorted(model_keys - desc_keys)
# group by prefix
by_prefix = {}
for m in extra_models:
    parts = m.rsplit('_', 1)
    prefix = parts[0] if len(parts) > 1 else m
    by_prefix.setdefault(prefix, []).append(m)

print('\nExtra model prefixes (first 10):')
for prefix in sorted(by_prefix.keys())[:10]:
    print(f'  {prefix}: {len(by_prefix[prefix])} items, e.g. {by_prefix[prefix][:2]}')

# Check if description keys follow a pattern like category_0001 vs category_test_0001
print('\nSample desc keys per category (first 3 per cat):')
desc_by_cat = {}
for k in sorted(desc_keys):
    # Extract category - everything before _test_ or before _NNNN
    if '_test_' in k:
        cat = k.split('_test_')[0]
    else:
        cat = '_'.join(k.split('_')[:-1])
    desc_by_cat.setdefault(cat, []).append(k)

for cat in sorted(desc_by_cat.keys())[:10]:
    print(f'  {cat}: {desc_by_cat[cat][:3]}')
