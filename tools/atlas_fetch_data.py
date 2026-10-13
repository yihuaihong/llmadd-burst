"""Fetch the ground-truth coordinates for the atlas-v2 landmark domain (manifold_domains.py) from the English
Wikipedia API (prop=coordinates, primary coordinate of each article) and write <DATA>/landmarks.csv
(name, title, lat, lon). Names = how the item is written in the prompt; title = the Wikipedia article.
The people and US-city domains read the Gurnee & Tegmark (2024) entity tables instead (MIT licence, fetched from
github.com/wesg52/world-models data/entity_datasets as <DATA>/gurnee_*.csv).

    python tools/atlas_fetch_data.py
"""

from __future__ import annotations

import csv
import json
import urllib.parse
import urllib.request

from manifold_domains import DATA

LANDMARKS = [  # (prompt name, Wikipedia title)
    ("Eiffel Tower", "Eiffel Tower"), ("Colosseum", "Colosseum"), ("Big Ben", "Big Ben"), ("Stonehenge", "Stonehenge"),
    ("Parthenon", "Parthenon"), ("Leaning Tower of Pisa", "Leaning Tower of Pisa"), ("Sagrada Familia", "Sagrada Família"),
    ("Brandenburg Gate", "Brandenburg Gate"), ("Neuschwanstein Castle", "Neuschwanstein Castle"), ("Louvre", "Louvre"),
    ("Buckingham Palace", "Buckingham Palace"), ("Tower Bridge", "Tower Bridge"), ("Edinburgh Castle", "Edinburgh Castle"),
    ("Alhambra", "Alhambra"), ("Hagia Sophia", "Hagia Sophia"), ("Saint Basil's Cathedral", "Saint Basil's Cathedral"),
    ("Matterhorn", "Matterhorn"), ("Mont Blanc", "Mont Blanc"), ("Mount Vesuvius", "Mount Vesuvius"), ("Atomium", "Atomium"),
    ("Charles Bridge", "Charles Bridge"), ("Mont-Saint-Michel", "Mont-Saint-Michel"), ("Trevi Fountain", "Trevi Fountain"),
    ("Great Pyramid of Giza", "Great Pyramid of Giza"), ("Abu Simbel", "Abu Simbel"), ("Victoria Falls", "Victoria Falls"),
    ("Table Mountain", "Table Mountain"), ("Mount Kilimanjaro", "Mount Kilimanjaro"), ("Mount Kenya", "Mount Kenya"),
    ("Taj Mahal", "Taj Mahal"), ("Great Wall of China", "Great Wall of China"), ("Forbidden City", "Forbidden City"),
    ("Mount Everest", "Mount Everest"), ("Mount Fuji", "Mount Fuji"), ("Angkor Wat", "Angkor Wat"),
    ("Petronas Towers", "Petronas Towers"), ("Burj Khalifa", "Burj Khalifa"), ("Petra", "Petra"),
    ("Potala Palace", "Potala Palace"), ("Terracotta Army", "Terracotta Army"), ("Golden Temple", "Golden Temple"),
    ("Western Wall", "Western Wall"), ("Shwedagon Pagoda", "Shwedagon Pagoda"), ("Tokyo Tower", "Tokyo Tower"),
    ("Borobudur", "Borobudur"), ("Marina Bay Sands", "Marina Bay Sands"), ("Kaaba", "Kaaba"), ("Persepolis", "Persepolis"),
    ("Mount Ararat", "Mount Ararat"), ("Ha Long Bay", "Ha Long Bay"), ("Taipei 101", "Taipei 101"),
    ("Statue of Liberty", "Statue of Liberty"), ("Golden Gate Bridge", "Golden Gate Bridge"), ("Grand Canyon", "Grand Canyon"),
    ("Mount Rushmore", "Mount Rushmore"), ("Empire State Building", "Empire State Building"), ("White House", "White House"),
    ("Niagara Falls", "Niagara Falls"), ("Space Needle", "Space Needle"), ("CN Tower", "CN Tower"),
    ("Chichen Itza", "Chichen Itza"), ("Teotihuacan", "Teotihuacan"), ("Machu Picchu", "Machu Picchu"),
    ("Christ the Redeemer", "Christ the Redeemer (statue)"), ("Iguazu Falls", "Iguazu Falls"), ("Aconcagua", "Aconcagua"),
    ("Panama Canal", "Panama Canal"), ("Hollywood Sign", "Hollywood Sign"), ("Gateway Arch", "Gateway Arch"),
    ("Hoover Dam", "Hoover Dam"), ("Denali", "Denali"), ("Lake Titicaca", "Lake Titicaca"),
    ("Sydney Opera House", "Sydney Opera House"), ("Uluru", "Uluru"), ("Sydney Harbour Bridge", "Sydney Harbour Bridge"),
    ("Aoraki", "Aoraki / Mount Cook"), ("Milford Sound", "Milford Sound"), ("Mount Erebus", "Mount Erebus"),
]


def coords(titles):
    q = urllib.parse.urlencode({"action": "query", "prop": "coordinates", "titles": "|".join(titles), "format": "json", "colimit": "max",
                                "redirects": 1})
    req = urllib.request.Request(f"https://en.wikipedia.org/w/api.php?{q}", headers={"User-Agent": "manifold-atlas/0.1"})
    d = json.load(urllib.request.urlopen(req, timeout=30))["query"]
    back = {}
    for m in d.get("normalized", []) + d.get("redirects", []):
        back[m["to"]] = back.get(m["from"], m["from"])
    out = {}
    for p in d["pages"].values():
        if "coordinates" in p:
            c = p["coordinates"][0]
            out[back.get(p["title"], p["title"])] = (c["lat"], c["lon"])
    return out


def main() -> None:
    got = {}
    for i in range(0, len(LANDMARKS), 40):
        got.update(coords([t for _, t in LANDMARKS[i:i + 40]]))
    rows = [(n, t, *got[t]) for n, t in LANDMARKS if t in got]
    missing = [t for _, t in LANDMARKS if t not in got]
    with open(DATA / "landmarks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "title", "lat", "lon"])
        w.writerows(rows)
    print(f"wrote {len(rows)} landmarks to {DATA / 'landmarks.csv'}; missing: {missing}")


if __name__ == "__main__":
    main()
