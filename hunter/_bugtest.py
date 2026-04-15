from data.products import find_product

tests = [
    ("iPad Air 11 M3 256 Starlight ESIM",        None,        "нет starlight 256 → тихо"),
    ("iPad Pro 11 (M5,2 TB,Wi-Fi) Black",         None,        "2TB нет → тихо"),
    ("iPhone 16 256 gb, pink",                    None,        "iPhone нет → тихо"),
    ("16 256 розовый",                            None,        "iPhone 16 нет → тихо"),
    ("MacBook Air 13 (M2/16Gb/256Gb) Midnight",   None,        "M2 нет → тихо"),
    ("MacBook Air 13 M2 256 Midnight",            None,        "M2 нет → тихо"),
    ("куплю MacBook Air 13 m4 16gb/512 gb",       "16/512",    "MW133 — 16GB/512GB"),
    ("iPad Air 11 M3 128 Starlight Wi Fi",        "starliight","starlight верно"),
    ("iPad Air 11 M3 128 Space Gray",             "space gray","space gray верно"),
    ("MacBook Air 13 m4 16/256 sky blue",         "sky blue",  "sky blue верно"),
    ("17 PRO MAX 1TB silver esim",                "1TB silver","17 pro max верно"),
    ("iPad 11 128 Silver WiFi",                   "silver",    "iPad 11 верно"),
    ("Apple Watch SE Gen 3 44mm Midnight",        "SE3 44",    "SE3 44 верно"),
    ("Air 11 M3 Wi-Fi 128Gb grey",                None,        "grey нет → тихо"),
]

print(f"{'#':<3} {'OK':<3} {'query':<48} {'got':<38} note")
print("-"*130)
for i, (q, expect, note) in enumerate(tests, 1):
    r = find_product(q)
    name = r["name"] if r else None
    if expect is None:
        ok = "OK" if name is None else "BUG"
    else:
        ok = "OK" if name and expect in name else "BUG"
    icon = "✅" if ok == "OK" else "❌"
    print(f"{i:<3} {icon:<3} {q[:48]!r:<50} {str(name):<38} {note}")
