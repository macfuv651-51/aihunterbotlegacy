import json, sys
import data.products as m

with open('data/keywords.json', encoding='utf-8') as f:
    data = json.load(f)

aliases = data.get('product_aliases', {})

ok = fail = 0
fails = []
total_estimated = sum(1 + len(v) for k, v in aliases.items()
                      if not k.startswith('_') and isinstance(v, list))

i = 0
for canonical, forms in aliases.items():
    if canonical.startswith('_') or not isinstance(forms, list):
        continue

    # canonical key resolves to itself
    norm = m._normalize(canonical)
    got = m._resolve_via_aliases(norm)
    i += 1
    if got == canonical:
        ok += 1
    else:
        fail += 1
        fails.append('  SELF  [%s]  got=[%s]  norm=%r' % (canonical, got, norm))

    # every alias resolves to canonical
    for form in forms:
        if not isinstance(form, str):
            continue
        norm = m._normalize(form)
        got = m._resolve_via_aliases(norm)
        i += 1
        if got == canonical:
            ok += 1
        else:
            fail += 1
            fails.append('  ALIAS [%s] <- %r  got=[%s]  norm=%r' % (canonical, form, got, norm))

    # progress every 50 canonicals
    sys.stdout.write(f'\r  {i}/{total_estimated}  OK={ok}  FAIL={fail}   ')
    sys.stdout.flush()

print(f'\n\nВсего кейсов: {ok + fail}')
print(f'OK:   {ok}')
print(f'FAIL: {fail}')
if fails:
    print('\nПровалы:')
    for line in fails:
        print(line)
else:
    print('\nВсе позиции прошли!')
