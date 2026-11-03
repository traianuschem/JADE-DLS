# Test fixtures

## alv_mono/
Auszug, unveraendert (byte-identisch), aus dem Repo-Zip
`Test+ReferenceData/example_files_monomodal.zip` (Ordner `example_files_monomodal/`, 52 ASC-Dateien).

Auswahlregel: 3 Winkel (30, 60, 90 Grad; Messungen 0053, 0056, 0059) x je 2 Wiederholungen
(`_0001`, `_0002`) = 6 Dateien, plus `080622_5_0053_averaged.ASC` (prueft, dass der Parser
`averaged`-Dateien herausfiltert). Gesamt ca. 290 kB.

Dateien:
- 080622_5_0053_0001.ASC, 080622_5_0053_0002.ASC, 080622_5_0053_averaged.ASC
- 080622_5_0056_0001.ASC, 080622_5_0056_0002.ASC
- 080622_5_0059_0001.ASC, 080622_5_0059_0002.ASC

`manifest.json` enthaelt Dateiname, Groesse und sha256; geprueft von `tests/test_fixtures_manifest.py`.
Nicht veraendern (Zeilenenden nicht konvertieren; `.gitattributes` ggf. beachten).
