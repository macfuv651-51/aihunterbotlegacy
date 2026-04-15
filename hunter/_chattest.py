"""
_chattest.py - проверяем конкретные запросы из чата.
"""
import sys
sys.path.insert(0, r"C:\Users\levk\Downloads\salerscipt")

from data.products import find_all_products, load_products

products_list = load_products()
prices = {p["name"]: p["price"] for p in products_list}

TESTS = [
    (
        "LTE запрос - у нас только wifi",
        "Новый не актив \n\niPad Pro 11 m5 512 LTE \nBlack \nSilver",
        None,
    ),
    (
        "Wi-Fi+Cellular - у нас только wifi",
        "iPad Air 11 М3 128GB Wi-Fi+Cellular \nПредложите цвет что есть",
        None,
    ),
    (
        "Wi-Fi + Cellular Space Gray - у нас только wifi",
        "Apple iPad Air 11 M3 (2025) 128 Gb Wi-Fi + Cellular Space Gray",
        None,
    ),
    (
        "32GB RAM / 512 Midnight - у нас нет 32GB",
        "MacBook Air 13 (2025) M4 (10CPU/10GPU/32GB/512GB) Midnight",
        None,
    ),
    (
        "Magic Mouse 4 USB-C черный (4 - неверное название юзера, у нас есть)",
        "Беспроводная мышь Apple Magic Mouse 4 USB-C Черный (MXK63)",
        "Magic Mouse USB-C black",
    ),
    (
        "MacBook Air 13 M4 16/512 Midnight - должен быть MW133",
        "куплю MacBook Air 13 m4 16gb/512 MIDNIGHT",
        "MW133 AIR 13 M4 16/512 midnight",
    ),
    (
        "MacBook Air 13 M4 24/512 Midnight - должен быть MC6C4",
        "MacBook Air 13 M4 24GB 512 midnight",
        "MC6C4 AIR 13 M4 24/512 midnight",
    ),
    (
        "iPad Air 11 M3 128 Starlight WiFi - должен матчить",
        "iPad Air 11 M3 128 Starlight Wi Fi",
        "iPad AIR 11 M3 128 wifi starlight",
    ),
    (
        "iPad 11 A16 128 Silver WiFi - должен матчить",
        "iPad 11 128 wifi silver",
        "iPad 11 A16 128 wifi silver",
    ),
    (
        "iPhone 16 256 pink - НЕТ в каталоге, не должен матчить iPad",
        "КУПЛЮ iPhone 16 256 gb, pink !!!",
        None,
    ),
    (
        "Watch SE3 40mm Midnight - должен матчить SE3",
        "Apple Watch SE Gen 3 40mm Midnight",
        "SE3 40 Midnight",
    ),
    (
        "Watch SE3 40mm + строка Midnight отдельно - НЕ должен матчить MacBook",
        "Apple Watch SE (Gen 3) 40mm \nMidnight",
        None,  # Expects NO match (multiline guard should block this)
    ),
]

PASS = 0
FAIL = 0

print(f"\n{'='*95}")
print(f"{'#':<4} {'OK':<4} {'ОЖИДАЛИ':<32} {'ПОЛУЧИЛИ':<36} ОПИСАНИЕ")
print(f"{'='*95}")

for i, (desc, query, expected_substr) in enumerate(TESTS, 1):
    found = find_all_products(query)
    names = [p["name"] for p in found]

    if expected_substr is None:
        ok = len(names) == 0
        got_str = "нет совпадений" if not names else names[0]
        exp_str = "нет совпадений"
    else:
        # Check if expected_substr is in ANY found product name (case-insensitive partial match)
        ok = any(expected_substr.lower() in n.lower() for n in names)
        got_str = names[0] if names else "нет совпадений"
        exp_str = expected_substr

    status = "OK" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1

    print(f"{i:<4} {status:<4} {exp_str[:31]:<32} {got_str[:35]:<36} {desc}")
    if found:
        for p in found:
            print(f"          Бот отправит: {p['name']} --- {p['price']} руб.")
    else:
        print(f"          Бот отправит: (ничего не отправит)")

print(f"{'='*95}")
print(f"Итог: {PASS}/{len(TESTS)} прошли, FAIL: {FAIL}")
