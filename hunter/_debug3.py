import sys; sys.path.insert(0, '.')
from data.products import _normalize, load_products, _storage_tokens, _model_nums

queries = [
    'air 13 m4 16 midnight',    # MC6J4 resolved canonical
    'air 13 m4 256 starlight',  # MC6L4 resolved canonical
    'air 15 m4 24 512 silver',  # air 15 m4 24/512 silver
    'air 13 m4 16 256 sky blue',# MC6T4
]

products = load_products()

for query in queries:
    qw = query.split()
    qs = _storage_tokens(qw)
    qm = _model_nums(qw)
    qset = set(qw)
    print(f"\n--- Query: '{query}' ---")
    print(f"  storage={qs}  model={qm}")

    for p in products:
        name = p['name']
        n = _normalize(name)
        nw = set(n.split())

        model_ok  = not qm or bool(qm & nw)
        storage_ok = not qs or bool(qs & nw)
        max_ok = ('max' in nw) == ('max' in qset)

        if not model_ok:
            continue
        if not storage_ok:
            continue
        if not max_ok:
            continue

        if query in n:
            print(f"  CONTAIN: {name}")
        else:
            overlap = sum(1 for w in qw if w in nw)
            score = overlap / len(qw)
            if score >= 0.75:
                matching = [w for w in qw if w in nw]
                print(f"  OVERLAP {score:.2f}: {name}  match={matching}")
