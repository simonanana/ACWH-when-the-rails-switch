"""
u01_icsid_reference.py — rebuild the ICSID Contracting State reference table from the
official ICSID/3 list (a verifiable source)

Source
------
ICSID/3 "List of Contracting States and Other Signatories of the Convention"
(as of June 9, 2020), https://icsid.worldbank.org/sites/default/files/ICSID-3.pdf
The table below is transcribed row by row from that document (signature date |
date of deposit of ratification | date of entry into force for the State), plus the
three denunciations in the document's footnotes (BOL / ECU / VEN) and the three
changes announced by ICSID after 2020-06-09 (ECU's renewed accession, KGZ, AGO).

The output schema is compatible with r26_eligibility_audit.load_ref /
legal_eligibility.EligibilityTable:
    iso3, state, signature, in_force_from, out_of_force_from, status_note, source, verified

Conventions for in_force_from / out_of_force_from (consistent with legal_eligibility.is_party):
    in_force_from empty              -> never a Contracting State (signed but not ratified / never signed)
    inf < out                        -> accession followed by denunciation (BOL, VEN)
    out < inf                        -> denunciation followed by renewed accession (ECU)

Usage
-----
    python src/u01_icsid_reference.py --out data/reference/icsid_convention_status_icsid3.csv
    # Row-by-row comparison with the existing reference table (strongly recommended):
    python src/u01_icsid_reference.py --out data/reference/icsid_convention_status_icsid3.csv \
        --compare data/reference/icsid_convention_status.csv
"""
from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path

import pandas as pd

# iso3|state|signature|deposit|entry_into_force   (empty field = has not occurred)
ICSID3_2020 = """\
AFG|Afghanistan|1966-09-30|1968-06-25|1968-07-25
ALB|Albania|1991-10-15|1991-10-15|1991-11-14
DZA|Algeria|1995-04-17|1996-02-21|1996-03-22
ARG|Argentina|1991-05-21|1994-10-19|1994-11-18
ARM|Armenia|1992-09-16|1992-09-16|1992-10-16
AUS|Australia|1975-03-24|1991-05-02|1991-06-01
AUT|Austria|1966-05-17|1971-05-25|1971-06-24
AZE|Azerbaijan|1992-09-18|1992-09-18|1992-10-18
BHS|Bahamas, The|1995-10-19|1995-10-19|1995-11-18
BHR|Bahrain|1995-09-22|1996-02-14|1996-03-15
BGD|Bangladesh|1979-11-20|1980-03-27|1980-04-26
BRB|Barbados|1981-05-13|1983-11-01|1983-12-01
BLR|Belarus|1992-07-10|1992-07-10|1992-08-09
BEL|Belgium|1965-12-15|1970-08-27|1970-09-26
BLZ|Belize|1986-12-19||
BEN|Benin|1965-09-10|1966-09-06|1966-10-14
BIH|Bosnia and Herzegovina|1997-04-25|1997-05-14|1997-06-13
BWA|Botswana|1970-01-15|1970-01-15|1970-02-14
BRN|Brunei Darussalam|2002-09-16|2002-09-16|2002-10-16
BGR|Bulgaria|2000-03-21|2001-04-13|2001-05-13
BFA|Burkina Faso|1965-09-16|1966-08-29|1966-10-14
BDI|Burundi|1967-02-17|1969-11-05|1969-12-05
CPV|Cabo Verde|2010-12-20|2010-12-27|2011-01-26
KHM|Cambodia|1993-11-05|2004-12-20|2005-01-19
CMR|Cameroon|1965-09-23|1967-01-03|1967-02-02
CAN|Canada|2006-12-15|2013-11-01|2013-12-01
CAF|Central African Republic|1965-08-26|1966-02-23|1966-10-14
TCD|Chad|1966-05-12|1966-08-29|1966-10-14
CHL|Chile|1991-01-25|1991-09-24|1991-10-24
CHN|China|1990-02-09|1993-01-07|1993-02-06
COL|Colombia|1993-05-18|1997-07-15|1997-08-14
COM|Comoros|1978-09-26|1978-11-07|1978-12-07
COD|Congo, Democratic Rep. of|1968-10-29|1970-04-29|1970-05-29
COG|Congo, Rep. of|1965-12-27|1966-06-23|1966-10-14
CRI|Costa Rica|1981-09-29|1993-04-27|1993-05-27
CIV|Cote d'Ivoire|1965-06-30|1966-02-16|1966-10-14
HRV|Croatia|1997-06-16|1998-09-22|1998-10-22
CYP|Cyprus|1966-03-09|1966-11-25|1966-12-25
CZE|Czech Republic|1993-03-23|1993-03-23|1993-04-22
DNK|Denmark|1965-10-11|1968-04-24|1968-05-24
DJI|Djibouti|2019-04-12|2020-06-09|2020-07-09
DOM|Dominican Republic|2000-03-20||
EGY|Egypt, Arab Rep. of|1972-02-11|1972-05-03|1972-06-02
SLV|El Salvador|1982-06-09|1984-03-06|1984-04-05
EST|Estonia|1992-06-23|1992-06-23|1992-07-23
SWZ|Eswatini|1970-11-03|1971-06-14|1971-07-14
ETH|Ethiopia|1965-09-21||
FJI|Fiji|1977-07-01|1977-08-11|1977-09-10
FIN|Finland|1967-07-14|1969-01-09|1969-02-08
FRA|France|1965-12-22|1967-08-21|1967-09-20
GAB|Gabon|1965-09-21|1966-04-04|1966-10-14
GMB|Gambia, The|1974-10-01|1974-12-27|1975-01-26
GEO|Georgia|1992-08-07|1992-08-07|1992-09-06
DEU|Germany|1966-01-27|1969-04-18|1969-05-18
GHA|Ghana|1965-11-26|1966-07-13|1966-10-14
GRC|Greece|1966-03-16|1969-04-21|1969-05-21
GRD|Grenada|1991-05-24|1991-05-24|1991-06-23
GTM|Guatemala|1995-11-09|2003-01-21|2003-02-20
GIN|Guinea|1968-08-27|1968-11-04|1968-12-04
GNB|Guinea-Bissau|1991-09-04||
GUY|Guyana|1969-07-03|1969-07-11|1969-08-10
HTI|Haiti|1985-01-30|2009-10-27|2009-11-26
HND|Honduras|1986-05-28|1989-02-14|1989-03-16
HUN|Hungary|1986-10-01|1987-02-04|1987-03-06
ISL|Iceland|1966-07-25|1966-07-25|1966-10-14
IDN|Indonesia|1968-02-16|1968-09-28|1968-10-28
IRQ|Iraq|2015-11-17|2015-11-17|2015-12-17
IRL|Ireland|1966-08-30|1981-04-07|1981-05-07
ISR|Israel|1980-06-16|1983-06-22|1983-07-22
ITA|Italy|1965-11-18|1971-03-29|1971-04-28
JAM|Jamaica|1965-06-23|1966-09-09|1966-10-14
JPN|Japan|1965-09-23|1967-08-17|1967-09-16
JOR|Jordan|1972-07-14|1972-10-30|1972-11-29
KAZ|Kazakhstan|1992-07-23|2000-09-21|2000-10-21
KEN|Kenya|1966-05-24|1967-01-03|1967-02-02
KOR|Korea, Rep. of|1966-04-18|1967-02-21|1967-03-23
XKX|Kosovo, Rep. of|2009-06-29|2009-06-29|2009-07-29
KWT|Kuwait|1978-02-09|1979-02-02|1979-03-04
KGZ|Kyrgyz Republic|1995-06-09||
LVA|Latvia|1997-08-08|1997-08-08|1997-09-07
LBN|Lebanon|2003-03-26|2003-03-26|2003-04-25
LSO|Lesotho|1968-09-19|1969-07-08|1969-08-07
LBR|Liberia|1965-09-03|1970-06-16|1970-07-16
LTU|Lithuania|1992-07-06|1992-07-06|1992-08-05
LUX|Luxembourg|1965-09-28|1970-07-30|1970-08-29
MDG|Madagascar|1966-06-01|1966-09-06|1966-10-14
MWI|Malawi|1966-06-09|1966-08-23|1966-10-14
MYS|Malaysia|1965-10-22|1966-08-08|1966-10-14
MLI|Mali|1976-04-09|1978-01-03|1978-02-02
MLT|Malta|2002-04-24|2003-11-03|2003-12-03
MRT|Mauritania|1965-07-30|1966-01-11|1966-10-14
MUS|Mauritius|1969-06-02|1969-06-02|1969-07-02
MEX|Mexico|2018-01-11|2018-07-27|2018-08-26
FSM|Micronesia, Federated States of|1993-06-24|1993-06-24|1993-07-24
MDA|Moldova|1992-08-12|2011-05-05|2011-06-04
MNG|Mongolia|1991-06-14|1991-06-14|1991-07-14
MNE|Montenegro|2012-07-19|2013-04-10|2013-05-10
MAR|Morocco|1965-10-11|1967-05-11|1967-06-10
MOZ|Mozambique|1995-04-04|1995-06-07|1995-07-07
NAM|Namibia|1998-10-26||
NRU|Nauru|2016-04-12|2016-04-12|2016-05-12
NPL|Nepal|1965-09-28|1969-01-07|1969-02-06
NLD|Netherlands|1966-05-25|1966-09-14|1966-10-14
NZL|New Zealand|1970-09-02|1980-04-02|1980-05-02
NIC|Nicaragua|1994-02-04|1995-03-20|1995-04-19
NER|Niger|1965-08-23|1966-11-14|1966-12-14
NGA|Nigeria|1965-07-13|1965-08-23|1966-10-14
MKD|North Macedonia|1998-09-16|1998-10-27|1998-11-26
NOR|Norway|1966-06-24|1967-08-16|1967-09-15
OMN|Oman|1995-05-05|1995-07-24|1995-08-23
PAK|Pakistan|1965-07-06|1966-09-15|1966-10-15
PAN|Panama|1995-11-22|1996-04-08|1996-05-08
PNG|Papua New Guinea|1978-10-20|1978-10-20|1978-11-19
PRY|Paraguay|1981-07-27|1983-01-07|1983-02-06
PER|Peru|1991-09-04|1993-08-09|1993-09-08
PHL|Philippines|1978-09-26|1978-11-17|1978-12-17
PRT|Portugal|1983-08-04|1984-07-02|1984-08-01
QAT|Qatar|2010-09-30|2010-12-21|2011-01-20
ROU|Romania|1974-09-06|1975-09-12|1975-10-12
RUS|Russian Federation|1992-06-16||
RWA|Rwanda|1978-04-21|1979-10-15|1979-11-14
WSM|Samoa|1978-02-03|1978-04-25|1978-05-25
SMR|San Marino|2014-04-11|2015-04-18|2015-05-18
STP|Sao Tome and Principe|1999-10-01|2013-05-20|2013-06-19
SAU|Saudi Arabia|1979-09-28|1980-05-08|1980-06-07
SEN|Senegal|1966-09-26|1967-04-21|1967-05-21
SRB|Serbia|2007-05-09|2007-05-09|2007-06-08
SYC|Seychelles|1978-02-16|1978-03-20|1978-04-19
SLE|Sierra Leone|1965-09-27|1966-08-02|1966-10-14
SGP|Singapore|1968-02-02|1968-10-14|1968-11-13
SVK|Slovak Republic|1993-09-27|1994-05-27|1994-06-26
SVN|Slovenia|1994-03-07|1994-03-07|1994-04-06
SLB|Solomon Islands|1979-11-12|1981-09-08|1981-10-08
SOM|Somalia|1965-09-27|1968-02-29|1968-03-30
SSD|South Sudan|2012-04-18|2012-04-18|2012-05-18
ESP|Spain|1994-03-21|1994-08-18|1994-09-17
LKA|Sri Lanka|1967-08-30|1967-10-12|1967-11-11
KNA|St. Kitts & Nevis|1994-10-14|1995-08-04|1995-09-03
LCA|St. Lucia|1984-06-04|1984-06-04|1984-07-04
VCT|St. Vincent and the Grenadines|2001-08-07|2002-12-16|2003-01-15
SDN|Sudan|1967-03-15|1973-04-09|1973-05-09
SWE|Sweden|1965-09-25|1966-12-29|1967-01-28
CHE|Switzerland|1967-09-22|1968-05-15|1968-06-14
SYR|Syria|2005-05-25|2006-01-25|2006-02-24
TZA|Tanzania|1992-01-10|1992-05-18|1992-06-17
THA|Thailand|1985-12-06||
TLS|Timor-Leste|2002-07-23|2002-07-23|2002-08-22
TGO|Togo|1966-01-24|1967-08-11|1967-09-10
TON|Tonga|1989-05-01|1990-03-21|1990-04-20
TTO|Trinidad and Tobago|1966-10-05|1967-01-03|1967-02-02
TUN|Tunisia|1965-05-05|1966-06-22|1966-10-14
TUR|Turkey|1987-06-24|1989-03-03|1989-04-02
TKM|Turkmenistan|1992-09-26|1992-09-26|1992-10-26
UGA|Uganda|1966-06-07|1966-06-07|1966-10-14
UKR|Ukraine|1998-04-03|2000-06-07|2000-07-07
ARE|United Arab Emirates|1981-12-23|1981-12-23|1982-01-22
GBR|United Kingdom|1965-05-26|1966-12-19|1967-01-18
USA|United States of America|1965-08-27|1966-06-10|1966-10-14
URY|Uruguay|1992-05-28|2000-08-09|2000-09-08
UZB|Uzbekistan|1994-03-17|1995-07-26|1995-08-25
YEM|Yemen, Republic of|1997-10-28|2004-10-21|2004-11-20
ZMB|Zambia|1970-06-17|1970-06-17|1970-07-17
ZWE|Zimbabwe|1991-03-25|1994-05-20|1994-06-19
"""

# Denouncing States from the footnotes + changes after 2020-06-09 (ICSID announcements)
# (iso3, state, signature, in_force_from, out_of_force_from, note)
AMENDMENTS = [
    ("BOL", "Bolivia", "1991-05-03", "1995-07-23", "2007-11-03",
     "ICSID/3 note: in force 1995-07-23; denunciation notice 2007-05-02, effective 2007-11-03"),
    ("VEN", "Venezuela", "1993-08-18", "1995-06-01", "2012-07-25",
     "ICSID/3 note: in force 1995-06-01; denunciation notice 2012-01-24, effective 2012-07-25"),
    # ECU: denunciation followed by renewed accession -> in_force_from holds the date of
    # re-entry into force, out_of_force_from the date the denunciation took effect
    ("ECU", "Ecuador", "1986-01-15", "2021-09-03", "2010-01-07",
     "ICSID/3 note: in force 1986-02-14; denunciation effective 2010-01-07; "
     "re-entry into force 2021-09-03 (ICSID announcement)"),
    ("KGZ", "Kyrgyz Republic", "1995-06-09", "2022-05-21", "",
     "ratification deposited 2022-04-21; in force 2022-05-21 (ICSID announcement)"),
    ("AGO", "Angola", "2022-07-14", "2022-10-21", "",
     "signed 2022-07-14; ratification deposited 2022-09-21; in force 2022-10-21"),
]

# Entities absent from ICSID/3 but present in the case file (never signed, or status undetermined)
NON_SIGNATORY_NOTES = {
    "BMU": ("Bermuda", "UK overseas territory; extension of the Convention to be "
            "checked against ICSID/8 — coded as NON-party pending verification", False),
    "HKG": ("Hong Kong SAR, China", "China has not notified extension; coded non-party", False),
    "MAC": ("Macao SAR, China", "China has not notified extension; coded non-party", False),
    "EUU": ("European Union", "REIO; cannot be a Contracting State (Art. 67)", True),
}


def build() -> pd.DataFrame:
    t = pd.read_csv(StringIO(ICSID3_2020), sep="|", header=None, dtype=str,
                    names=["iso3", "state", "signature", "deposit", "eif"]).fillna("")
    rows = []
    for _, r in t.iterrows():
        rows.append({"iso3": r.iso3, "state": r.state, "signature": r.signature,
                     "in_force_from": r.eif, "out_of_force_from": "",
                     "status_note": ("signed, not ratified (as of 2020-06-09)"
                                     if not r.eif else "Contracting State"),
                     "source": "ICSID/3 as of 2020-06-09", "verified": "TRUE"})
    ref = pd.DataFrame(rows).set_index("iso3")
    for iso, st, sig, inf, out, note in AMENDMENTS:
        ref.loc[iso] = {"state": st, "signature": sig, "in_force_from": inf,
                        "out_of_force_from": out, "status_note": note,
                        "source": "ICSID/3 note or ICSID announcement",
                        "verified": "TRUE"}
    for iso, (st, note, ver) in NON_SIGNATORY_NOTES.items():
        if iso not in ref.index:
            ref.loc[iso] = {"state": st, "signature": "", "in_force_from": "",
                            "out_of_force_from": "", "status_note": note,
                            "source": "not listed in ICSID/3",
                            "verified": "TRUE" if ver else "FALSE"}
    ref = ref.reset_index().sort_values("iso3")
    # Self-check: key dates
    chk = ref.set_index("iso3")
    assert chk.loc["VEN", "out_of_force_from"] == "2012-07-25"
    assert chk.loc["IRQ", "in_force_from"] == "2015-12-17"
    assert chk.loc["MEX", "in_force_from"] == "2018-08-26"
    assert chk.loc["CAN", "in_force_from"] == "2013-12-01"
    assert chk.loc["RUS", "in_force_from"] == ""
    return ref


def compare(new: pd.DataFrame, old_path: str) -> pd.DataFrame:
    old = pd.read_csv(old_path, dtype=str).fillna("")
    cols = ["in_force_from", "out_of_force_from"]
    for c in cols:
        old[c] = pd.to_datetime(old[c].replace("", None), errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    m = new.merge(old[["iso3"] + cols], on="iso3", how="outer",
                  suffixes=("_icsid3", "_yours"), indicator=True)
    diff = m[(m["_merge"] != "both") |
             (m.in_force_from_icsid3 != m.in_force_from_yours) |
             (m.out_of_force_from_icsid3 != m.out_of_force_from_yours)]
    return diff


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/reference/icsid_convention_status_icsid3.csv")
    ap.add_argument("--compare", default=None, help="path to the existing reference table")
    a = ap.parse_args()
    ref = build()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    ref.to_csv(a.out, index=False)
    n_party = (ref.in_force_from != "").sum()
    print(f"[u01] {len(ref)} rows, of which current or former Contracting States: {n_party} -> {a.out}")
    if a.compare:
        d = compare(ref, a.compare)
        p = Path(a.out).with_name("diff_icsid_reference.csv")
        d.to_csv(p, index=False)
        print(f"[u01] rows differing from {a.compare}: {len(d)} -> {p}")
        if len(d):
            print(d.to_string(index=False))


if __name__ == "__main__":
    main()
