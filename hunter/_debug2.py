import sys, json; sys.path.insert(0, '.')
from data.products import load_products, _normalize

products = load_products()
print('=== MacBook Air in catalog ===')
for p in products:
    n = p['name']
    if 'air' in n.lower() or 'AIR' in n:
        print(f'  {n}  =>  {_normalize(n)}')

print()
with open('data/keywords.json', encoding='utf-8') as f:
    kw = json.load(f)
aliases = kw.get('product_aliases', {})
print('=== Air M4 canonicals in keywords.json ===')
for k in sorted(aliases.keys()):
    if 'air' in k and 'm4' in k:
        forms = aliases[k]
        print(f'  [{k}]')
        for form in (forms if isinstance(forms, list) else []):
            print(f'    - {form}')
