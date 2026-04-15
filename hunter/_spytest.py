# Тестируем именно те запросы из spy_log которые были неправильными
from data.products import find_product, find_all_products

tests = [
    # (описание, строка запроса, ожидаемый результат или None)
    ("Starlight → не purple",
     "iPad Air 11 M3 256 Starlight ESIM",
     None),  # starlight 256 нет в каталоге

    ("(M5,2 TB) → не должен давать 512",
     "iPad Pro 11 (M5,2 TB,Wi-Fi) Black",
     None),  # 2TB нет в каталоге

    ("iPhone 16 → не iPad",
     "iPhone 16 256 gb, pink",
     None),  # iPhone нет в каталоге

    ("16 256 розовый → не iPad (это iPhone)",
     "16 256 розовый",
     None),  # iPhone нет в каталоге

    ("17 pro max + 16 256 pink → только iPad если он там есть",
     "16 256 pink",
     None),  # iPhone нет в каталоге

    ("grey → не starlight",
     "Air 11 M3 (2025) Wi-Fi 128Gb grey",
     None),  # grey нет в каталоге

    ("M2 MacBook → не M4 продукт",
     "MacBook Air 13 (M2/16Gb/256Gb) Midnight",
     None),  # M2 нет в каталоге

    ("КУПЛЮ iPhone 16 → не iPad",
     "iPhone 16 256 gb, pink",
     None),  # iPhone нет в каталоге

    ("16 256 Pink отдельной строкой",
     "16 256 Pink",
     None),  # iPhone нет в каталоге

    ("MacBook Air M4 16gb/512 MIDNIGHT → MW133 не MC6C4",
     "MacBook Air 13 m4 16gb/512 gb MIDNIGHT",
     "MW133"),  # MW133 = 16/512 midnight

    # Проверяем что правильные запросы не сломались
    ("iPad 11 A16 128 silver — должен работать",
     "iPad 11 128 wifi silver",
     "A16 128 wifi silver"),

    ("iPad Air 11 M3 128 Starlight — должен работать",
     "iPad Air 11 M3 128 Starlight Wi Fi",
     "starliight"),

    ("iPad Air 11 M3 128 Space Gray — должен работать",
     "iPad Air 11 128GB Space Gray Wi-Fi M3",
     "space gray"),

    ("MacBook Air 13 M4 16/256 sky blue — должен работать",
     "MacBook Air 13 m4 16/256 sky blue",
     "sky blue"),

    ("iPad PRO 11 M5 512 black — должен работать",
     "iPad Pro 11 m5 512gb black wi-fi",
     "M5 512 wifi black"),

    ("SE3 44 midnight — должен работать",
     "Apple Watch SE Gen 3 44mm Midnight",
     "SE3 44 midnight"),

    ("MC6C4 24GB 512 midnight — должен работать (24GB явно указано)",
     "MacBook Air 13 M4 24GB 512 midnight",
     "MC6C4"),
]

print(f"{'#':<3} {'OK':<4} {'ожидаем':<10} {'получили':<40} запрос")
print("=" * 120)
bugs = 0
for i, (desc, query, expect) in enumerate(tests, 1):
    r = find_product(query)
    name = r["name"] if r else None
    if expect is None:
        ok = name is None
    else:
        ok = name is not None and expect in name
    icon = "✅" if ok else "❌"
    if not ok:
        bugs += 1
    exp_str = "None" if expect is None else expect
    print(f"{i:<3} {icon:<4} {exp_str:<10} {str(name):<40} {query[:60]!r}")

print()
print(f"Итог: {len(tests)-bugs}/{len(tests)} прошли, ❌ багов: {bugs}")
