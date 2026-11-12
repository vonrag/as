"""Данные игроков для серии карточек-легенд. render.py берёт игрока по ключу: --player zidane|ronaldo.

Надписи Зидана записаны по-русски (английская версия — через словарь EN в render.py),
надписи Роналдо — сразу по-английски.
Рейтинги карточек — придуманные «игровые», не официальные цифры.
"""
import numpy as np


def flag_france(w, h):
    a = np.zeros((h, w, 3), np.float32)
    a[:, : w // 3] = (30, 60, 170)
    a[:, w // 3: 2 * w // 3] = (245, 245, 245)
    a[:, 2 * w // 3:] = (225, 40, 55)
    return a


def flag_brazil(w, h):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    a = np.zeros((h, w, 3), np.float32)
    a[:] = (0, 150, 60)
    cx, cy = w / 2, h / 2
    rhomb = np.abs(xx - cx) / (w * 0.42) + np.abs(yy - cy) / (h * 0.40) <= 1
    a[rhomb] = (254, 210, 0)
    circ = np.hypot(xx - cx, yy - cy) <= h * 0.22
    a[circ] = (20, 40, 130)
    band = circ & (np.abs((yy - cy) - 0.18 * (xx - cx) + h * 0.02) < h * 0.025)
    a[band] = (245, 245, 245)
    return a


def flag_england(w, h):
    a = np.full((h, w, 3), 245, np.float32)
    t = int(h * 0.2)
    a[h // 2 - t // 2: h // 2 + t // 2, :] = (205, 20, 45)
    a[:, w // 2 - t // 2: w // 2 + t // 2] = (205, 20, 45)
    return a


ZIDANE = {
    "out": "zidane_card",
    "flag": flag_france,
    "country": "ФРАНЦИЯ",
    "country_glow": (80, 120, 255),
    "country_tint": (30, 55, 140),
    "pos": "ЦАП",
    "pos_long": "АТАКУЮЩИЙ ПОЛУЗАЩИТНИК",
    "club_lines": [("РЕАЛ", 230, 800), ("МАДРИД", 170, 1010)],
    "club_glow": (170, 130, 255),
    "club_tint": (90, 70, 150),
    "club_sub": "ЮВЕНТУС · РЕАЛ · СБОРНАЯ ФРАНЦИИ",
    "rating": "96",
    "card_name": "ZIDANE",
    "back_letter": "Z",
    "full_name": "ЗИНЕДИН ЗИДАН",
    "nickname": "ZIZOU",
    "jersey": {"body": (26, 44, 112), "trim": (255, 255, 255), "num": (255, 255, 255),
               "num_stroke": (200, 160, 70), "stripe": [(30, 70, 170), (255, 255, 255), (220, 40, 50)],
               "number": "10"},
    "card_stats": [("85", "СКР"), ("91", "УДР"), ("95", "ПАС"), ("97", "ДРБ"), ("76", "ЗЩТ"), ("88", "ФИЗ")],
    "stats": [("СКОРОСТЬ", 85), ("УДАРЫ", 91), ("ПАСЫ", 95), ("ДРИБЛИНГ", 97), ("ЗАЩИТА", 76), ("ФИЗИКА", 88)],
    "clubs": [
        ("1989–1992", "КАНН", "61", "6"),
        ("1992–1996", "БОРДО", "139", "28"),
        ("1996–2001", "ЮВЕНТУС", "151", "24"),
        ("2001–2006", "РЕАЛ МАДРИД", "155", "37"),
    ],
    "national": "СБОРНАЯ ФРАНЦИИ: 108 МАТЧЕЙ · 31 ГОЛ",
    "trophies_ru": [
        ("1998", "ЧЕМПИОН", "МИРА"),
        ("2000", "ЧЕМПИОН", "ЕВРОПЫ"),
        ("2002", "ЛИГА", "ЧЕМПИОНОВ"),
        ("1998", "ЗОЛОТОЙ", "МЯЧ"),
        ("×3", "ИГРОК ГОДА", "ФИФА"),
        ("×2", "ЧЕМПИОН", "ИТАЛИИ"),
    ],
    "trophies_en": [
        ("1998", "WORLD CUP", "WINNER"),
        ("2000", "EURO", "WINNER"),
        ("2002", "CHAMPIONS", "LEAGUE"),
        ("1998", "BALLON", "D'OR"),
        ("×3", "FIFA WORLD", "PLAYER"),
        ("×2", "SERIE A", "CHAMPION"),
    ],
}

RONALDO = {
    "out": "ronaldo_card",
    "flag": flag_brazil,
    "country": "BRAZIL",
    "country_glow": (60, 200, 90),
    "country_tint": (25, 110, 50),
    "pos": "ST",
    "pos_long": "STRIKER",
    "club_lines": [("INTER", 250, 900)],
    "club_glow": (70, 120, 255),
    "club_tint": (30, 60, 150),
    "club_sub": "BARCELONA · INTER · REAL MADRID · BRAZIL",
    "rating": "96",
    "card_name": "RONALDO",
    "back_letter": "R",
    "full_name": "RONALDO NAZÁRIO",
    "nickname": "R9",
    "jersey": {"body": (250, 214, 30), "trim": (0, 140, 60), "num": (0, 120, 55),
               "num_stroke": (255, 255, 255), "stripe": [(0, 140, 60), (20, 40, 130), (0, 140, 60)],
               "number": "9"},
    "card_stats": [("97", "PAC"), ("95", "SHO"), ("81", "PAS"), ("95", "DRI"), ("45", "DEF"), ("82", "PHY")],
    "stats": [("PACE", 97), ("SHOOTING", 95), ("PASSING", 81), ("DRIBBLING", 95), ("DEFENDING", 45), ("PHYSICAL", 82)],
    "clubs": [
        ("1994–1996", "PSV", "46", "42"),
        ("1996–1997", "BARCELONA", "37", "34"),
        ("1997–2002", "INTER", "68", "49"),
        ("2002–2007", "REAL MADRID", "127", "83"),
    ],
    "national": "BRAZIL: 98 CAPS · 62 GOALS",
    "trophies_ru": None,
    "trophies_en": [
        ("×2", "WORLD CUP", "WINNER"),
        ("×2", "BALLON", "D'OR"),
        ("×3", "FIFA WORLD", "PLAYER"),
        ("2002", "WORLD CUP", "GOLDEN BOOT"),
        ("1998", "UEFA CUP", "WINNER"),
        ("×2", "LA LIGA", "CHAMPION"),
    ],
}

KAKA = {
    "out": "kaka_card",
    "flag": flag_brazil,
    "country": "BRAZIL",
    "country_glow": (60, 200, 90),
    "country_tint": (25, 110, 50),
    "pos": "CAM",
    "pos_long": "ATTACKING MIDFIELDER",
    "club_lines": [("MILAN", 250, 900)],
    "club_glow": (235, 50, 60),
    "club_tint": (135, 28, 34),
    "club_sub": "SÃO PAULO · MILAN · REAL MADRID · BRAZIL",
    "rating": "95",
    "card_name": "KAKÁ",
    "back_letter": "K",
    "full_name": "RICARDO KAKÁ",
    "nickname": "KAKÁ",
    "jersey": {"body": (200, 20, 32), "trim": (15, 15, 15), "num": (255, 255, 255),
               "num_stroke": (15, 15, 15), "stripe": None, "stripes": [(200, 20, 32), (18, 18, 18)],
               "number": "22"},
    "card_stats": [("93", "PAC"), ("89", "SHO"), ("90", "PAS"), ("93", "DRI"), ("45", "DEF"), ("81", "PHY")],
    "stats": [("PACE", 93), ("SHOOTING", 89), ("PASSING", 90), ("DRIBBLING", 93), ("DEFENDING", 45), ("PHYSICAL", 81)],
    "clubs": [
        ("2001–2003", "SÃO PAULO", "59", "23"),
        ("2003–2009", "MILAN", "193", "70"),
        ("2009–2013", "REAL MADRID", "85", "23"),
        ("2015–2017", "ORLANDO CITY", "75", "24"),
    ],
    "national": "BRAZIL: 92 CAPS · 29 GOALS",
    "trophies_ru": None,
    "trophies_en": [
        ("2002", "WORLD CUP", "WINNER"),
        ("2007", "BALLON", "D'OR"),
        ("2007", "FIFA WORLD", "PLAYER"),
        ("2007", "CHAMPIONS", "LEAGUE"),
        ("2004", "SERIE A", "CHAMPION"),
        ("×2", "CONFED CUP", "WINNER"),
    ],
}

BECKHAM = {
    "out": "beckham_card",
    "flag": flag_england,
    "country": "ENGLAND",
    "country_glow": (230, 60, 80),
    "country_tint": (120, 30, 45),
    "pos": "RM",
    "pos_long": "RIGHT MIDFIELDER",
    "club_lines": [("MANCHESTER", 130, 790), ("UNITED", 210, 990)],
    "club_glow": (235, 40, 50),
    "club_tint": (140, 22, 30),
    "club_sub": "MAN UTD · REAL MADRID · LA GALAXY · ENGLAND",
    "rating": "93",
    "card_name": "BECKHAM",
    "back_letter": "B",
    "full_name": "DAVID BECKHAM",
    "nickname": "BECKS",
    "jersey": {"body": (205, 22, 32), "trim": (255, 255, 255), "num": (255, 255, 255),
               "num_stroke": (20, 20, 20), "stripe": None, "number": "7"},
    "card_stats": [("81", "PAC"), ("87", "SHO"), ("95", "PAS"), ("85", "DRI"), ("66", "DEF"), ("79", "PHY")],
    "stats": [("PACE", 81), ("SHOOTING", 87), ("PASSING", 95), ("DRIBBLING", 85), ("DEFENDING", 66), ("PHYSICAL", 79)],
    "clubs": [
        ("1992–2003", "MAN UTD", "265", "62"),
        ("2003–2007", "REAL MADRID", "116", "13"),
        ("2007–2012", "LA GALAXY", "98", "18"),
        ("2009–2010", "MILAN (LOAN)", "29", "2"),
    ],
    "national": "ENGLAND: 115 CAPS · 17 GOALS",
    "trophies_ru": None,
    "trophies_en": [
        ("1999", "CHAMPIONS", "LEAGUE"),
        ("×6", "PREMIER", "LEAGUE"),
        ("×2", "FA CUP", "WINNER"),
        ("2007", "LA LIGA", "CHAMPION"),
        ("×2", "MLS CUP", "WINNER"),
        ("2013", "LIGUE 1", "CHAMPION"),
    ],
    "ru": {
        "country": "АНГЛИЯ",
        "pos": "ПП",
        "pos_long": "ПРАВЫЙ ПОЛУЗАЩИТНИК",
        "club_lines": [("МАНЧЕСТЕР", 130, 790), ("ЮНАЙТЕД", 170, 990)],
        "club_sub": "МЮ · РЕАЛ · ЛА ГЭЛАКСИ · АНГЛИЯ",
        "full_name": "ДЭВИД БЕКХЭМ",
        "card_stats": [("81", "СКР"), ("87", "УДР"), ("95", "ПАС"), ("85", "ДРБ"), ("66", "ЗЩТ"), ("79", "ФИЗ")],
        "stats": [("СКОРОСТЬ", 81), ("УДАРЫ", 87), ("ПАСЫ", 95), ("ДРИБЛИНГ", 85), ("ЗАЩИТА", 66), ("ФИЗИКА", 79)],
        "clubs": [
            ("1992–2003", "МАН ЮНАЙТЕД", "265", "62"),
            ("2003–2007", "РЕАЛ МАДРИД", "116", "13"),
            ("2007–2012", "ЛА ГЭЛАКСИ", "98", "18"),
            ("2009–2010", "МИЛАН (АРЕНДА)", "29", "2"),
        ],
        "national": "СБОРНАЯ АНГЛИИ: 115 МАТЧЕЙ · 17 ГОЛОВ",
        "trophies_ru": [
            ("1999", "ЛИГА", "ЧЕМПИОНОВ"),
            ("×6", "ЧЕМПИОН", "АНГЛИИ"),
            ("×2", "КУБОК", "АНГЛИИ"),
            ("2007", "ЧЕМПИОН", "ИСПАНИИ"),
            ("×2", "КУБОК", "MLS"),
            ("2013", "ЧЕМПИОН", "ФРАНЦИИ"),
        ],
    },
}

PLAYERS = {"zidane": ZIDANE, "ronaldo": RONALDO, "kaka": KAKA, "beckham": BECKHAM}
