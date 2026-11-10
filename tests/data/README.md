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

## synthetic_golden/
Referenzergebnisse der Synthetik-Studie (10 Proben, Replikat 0; Ground Truth, Settings, Ergebnisse der Methoden
B, C2, C4, D, NNLS, Regularized und Tabellen je Datei). Die Referenz wurde mit den unveraenderten JADE-DLS-Modulen
berechnet; die Eingabedaten werden von `ade_dls.benchmark` aus den Seeds neu erzeugt.
`golden.json.sha256` sichert die Integritaet, `make_golden.py` dokumentiert die Extraktion, `NOTICE.md` die Lizenz
(Daten: CC BY-SA 4.0; der Simulator selbst steht unter GPL wie ADE). Nicht von Hand aendern; geprueft von `tests/test_benchmark_golden.py`.
