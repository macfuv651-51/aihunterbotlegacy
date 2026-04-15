import sys, json
sys.path.insert(0, '.')
from data.products import load_products, find_all_products

# === 1. Полный каталог ===
products = load_products()
print(f"Всего товаров в каталоге: {len(products)}\n")
print("=" * 80)
print("ЧТО ЕСТЬ В products.json (это всё что бот может отвечать):")
print("=" * 80)
prev_brand = None
prev_cat = None
for p in products:
    brand = p.get('brand', '')
    cat = p.get('category', '')
    avail = "🟢" if p.get('available') else "🔴"
    price = p['price'].rstrip('0').rstrip('.')
    if brand != prev_brand:
        print(f"\n  {brand}")
        prev_brand = brand
        prev_cat = None
    if cat != prev_cat:
        print(f"    {cat}")
        prev_cat = cat
    print(f"      {avail} {p['name']} — {price}")

# === 2. Симуляция по запросам из чата ===
messages = [
    ('Арут',         '17 Pro 1TB Silver esim'),
    ('Technoperry',  '17 pro max 256 silver 1sim'),
    ('Andrey',       '17 pro max 256 blue eSIM'),
    ('Евгений',      '17 pro max 1Tb blue 1sim'),
    ('Владимир',     'Apple Mac mini M4 16/256 Silver'),
    ('Egor',         '17 Air 256 Gold E-sim'),
    ('👑',           '17 Pro Max 1Tb Silver esim'),
    ('Zakupka.AJ',   '17 256GB Lavender (eSIM + nano-SIM)'),
    ('Валек',        'Apple iPhone 17 Pro Max 256GB Deep Blue eSIM'),
    ('Support',      'mde04'),
    ('Амина',        '17 pro max 512 orange 1 sim'),
    ('316рубин',     '17 pro max 256 silver esim'),
    ('316рубин',     '17 Pro Max 1Tb Orange eSIM'),
    ('Мистер А',     '17 512 black 1Sim'),
    ('Slava',        '17 Pro Max 1TB Orange (eSim)'),
    ('Urus',         's11 42 Silver SB S/M'),
    ('Kate',         '17 про Макс 1 тб синий 1 сим'),
    ('TECHNOV',      'MC6J4'),
    ('TECHNOV',      'MC6L4'),
    ('TECHNOV',      'AW S10, 42 Silver Sport Loop Blue Cloud'),
    ('Дарья',        'Macbook Air 15 M4 16/512Gb Silver MW1H3'),
    ('Алекс',        '17 256 Sage 1sim'),
    ('Алекс',        '17 Pro Max 256 Blue 1sim'),
    ('Алекс',        '17 Pro Max 256 Silver 1sim'),
    ('Egor',         'Apple iPhone 17 Pro 256 Silver (eSim)'),
    ('Stasya',       '17 256 sage 1 sim'),
    ('Urus',         '17 Pro Max 1TB Silver 1sim'),
    ('Zakupka',      '17 Pro Max 512Gb Deep Blue (eSim + nano-Sim)'),
    ('Султан',       'iPad Air 11 M3 (2025) 512GB Wi-Fi Starlight'),
    ('Лев',          'iPad Air 13 M3 128 Gb Wi-Fi Purple'),
    ('Lev_К',        'iPhone 17 Pro Max 256GB Silver nanoSIM + eSIM'),
    ('Red Price',    '17 pro 512 silver esim'),
    ('Васяо',        'air 13 m4 16/256 blue'),
    ('Васяо',        'air 15 m4 24/512 silver'),
    ('VADIM',        'iPad Air 11 M3 128 Gb Wi-Fi Purple'),
    ('Alina',        '17 про макс 1тб оранж Есим'),
    ('D777',         'iPhone 16e 512GB White SIM+eSIM'),
    ('РубиСтор',     'iPhone 17 Pro, 256 синий eSim'),
    ('Support',      'mc6j4'),
    ('Support',      'mc6l4'),
    ('Кирилл',       'MacBook Air 13 M4 2025 Midnight MW123'),
    ('Kate',         '17 про 256 синий 1 сим'),
    ('Сергей',       'MacBook Air 15 Silver Early 2025 MW1G3'),
    ('SOLAR',        '17 Pro Max 512GB Silver esim'),
    ('Howard',       '17 pro max 512 blue 1sim'),
    ('Howard',       '17 pro max 512 silver 1sim'),
    ('Юлия',         '17 Pro Max 256 Blue esim'),
    ('Technoperry',  '17 256 lavender 1sim'),
    ('ВСЯТЕХНИКА',   '17 Pro Max 1TB Orange'),
    ('TDeus',        'Apple iPad Air 11 2025 M3 Wi-Fi 128Gb Starlight'),
    ('Владимир',     '17 Pro Max 1TB Cosmic Orange 1Sim'),
    ('Urus',         '17 Pro 512Gb Deep Blue'),
    ('VADIM',        'iPad Air 11 M3 128 Gb Wi-Fi Purple'),
    ('Stasya',       '17 pro max 256 orange 1 sim'),
    ('Арут',         '17 Pro 1TB Silver esim'),
    ('Urus',         '17 Pro Max 1TB Silver 1sim'),
]

print("\n\n" + "=" * 80)
print("СИМУЛЯЦИЯ: что бот ответит на каждое сообщение")
print("=" * 80)

for sender, msg in messages:
    results = find_all_products(msg)
    if results:
        reply = ' | '.join(f"{r['name']} — {r['price'].rstrip('0').rstrip('.')}" for r in results)
        status = "✅ ОТВЕТИТ"
    else:
        reply = "— НЕТ ТОВАРА В ПРАЙСЕ"
        status = "🔇 МОЛЧИТ"
    print(f"\n{status} [{sender}]")
    print(f"  Запрос: {msg}")
    print(f"  Ответ:  {reply}")
