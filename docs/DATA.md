# Data: retrieval, retained captures, labels

## 1. CSE-CIC-IDS2018

The dataset is published by the Canadian Institute for Cybersecurity and the Communications
Security Establishment in a public Amazon S3 bucket
(`https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/`, dataset page
<https://www.unb.ca/cic/datasets/ids-2018.html>). No account is needed.

The raw traffic is distributed as one archive of per-host captures per day:

| Day | Object | Size (bytes) |
|---|---|---|
| Wednesday-14-02-2018 | `…/pcap.zip` | 39,913,353,098 |
| Thursday-15-02-2018 | `…/pcap.zip` | 41,283,382,768 |
| Friday-16-02-2018 | `…/pcap.zip` | 38,535,667,707 |
| Tuesday-20-02-2018 | `…/pcap.rar` | 44,391,980,380 |
| Wednesday-21-02-2018 | `…/pcap.zip` | 53,462,820,707 |
| Thursday-22-02-2018 | `…/pcap.zip` | 50,240,938,251 |
| Friday-23-02-2018 | `…/pcap.zip` | 59,093,922,810 |
| Wednesday-28-02-2018 | `…/pcap.zip` | 53,251,694,487 |
| Thursday-01-03-2018 | `…/pcap.zip` | 52,358,069,106 |
| Friday-02-03-2018 | `…/pcap.zip` | 44,789,835,888 |

(prefix `Original Network Traffic and Log data/<day>/`; listing read on 9 October 2026; sizes and
ETags are pinned in `configs/s3_inventory_ids2018.csv`, and `fetch` compares the live listing
with the pinned one and writes `inventory_check.csv`).

### Which captures are retained

The corpus is defined by a rule, not by a hand-made list:

1. **Days.** The nine days distributed as ZIP archives. `Tuesday-20-02-2018` is the only day
   distributed as a RAR archive; RAR does not allow single members to be retrieved with range
   requests, so the day is excluded by default (`corpus.excluded_days`). To include it, download
   and unpack the archive, add the day to `corpus.days` and use `--local-root`.
2. **Victim captures.** Every capture whose file name carries the address of a host attacked on
   that day (`victims_internal` in `configs/attack_schedule_ids2018.yaml`).
3. **Background captures.** `corpus.background_hosts_per_day` (20) further captures per day,
   drawn uniformly at random without replacement from the remaining captures sorted by name,
   with `numpy.random.default_rng(corpus.seed + index of the day)`; `corpus.seed = 42`.
   Captures larger than `corpus.max_background_member_bytes` are not drawn.

ZIP archives keep their directory at the end of the file, so the member list and single
members are read with HTTP range requests (`twws/fetch.py`): only the selected captures are
transferred. `fetch --list-only` prints what would be retrieved and how large it is.

### Manifest

`<work_dir>/data/manifest.csv` has one row per retained capture:

`dataset, day, archive_key, archive_size, archive_etag, member, role (victim|background),
host_ip, file_size, crc32 (from the archive directory), sha256 (of the retrieved file), local_path`

`archive_members.csv` lists *every* member of every archive with its role
(`victim`, `background`, `unselected`), so the selection can be audited. Both files are copied
to the supplementary material by `report`.

The S3 ETag of a multipart upload is not an MD5 of the object; `twws.utils.s3_multipart_etag`
recomputes it for a locally stored archive (part size 16 MiB) if a full archive is to be verified.

### If the archives are already on disk

```bash
python -m twws.cli ingest --config configs/paper.yaml --local-root /data/ids2018
```

accepts `<root>/<day>/pcap.zip` (the same selection is applied to the local archive) or
`<root>/<day>/<capture files>` (captures already unpacked).

### Capture check and repair

Some captures are damaged in the archives as distributed (the CRC-32 of the retrieved member
matches the one in the archive directory, so the damage is not a transfer error): the last packet
record is incomplete, or a few hundred stray bytes sit between two records. libpcap stops at the
first such place; Snort then exits with an error, and Zeek stops with a fatal error and does not
write the connections that were still open (more than half of the flows of some captures).

Before Zeek and Snort read a capture, `twws/pcapcheck.py` walks its record chain. A record is
accepted when its header is plausible (time stamp between one day before and three days after the
start of the capture, sub-second field in range, captured length not above the snapshot length
and not above the original length). Where the chain breaks, the bytes up to the next offset at
which 16 consecutive plausible records start are skipped (the approach of pcapfix), and the file
is rewritten from the valid records; when only the tail is damaged, the file is cut at the end of
the last complete record. pcapng files are only checked and cut. The outcome is written next to
the capture (`<capture>.check.json`: records, skipped regions and bytes, size and SHA-256 before
the repair) and collected in `extract/capture_checks.csv`, which `report` copies to the
supplementary material; the manuscript reports the number of repaired captures and the bytes
removed. Zeek is run strictly: a capture on which Zeek fails stops the extraction instead of
contributing a partial flow table.

### Flow extraction

`zeek -C -b -r <capture> zeek/twws_flows.zeek TWWS::entropy_max_bytes=4096`

* `-b` (bare mode): only the scripts loaded by `twws_flows.zeek` run; the list of loaded scripts
  is kept per capture (`loaded_scripts.log`).
* One Zeek connection becomes two unidirectional flow records (a direction without packets
  yields no record).
* Per direction: packets, bytes, TCP flag counts, payload bytes from the headers and payload
  bytes actually captured, first/last packet time, Shannon entropy of the first 4096 payload
  bytes (missing when there is no payload), TLS handshake seen, leaf certificate self-signed
  (unknown under TLS 1.3, where the certificate is encrypted).
* A connection between two captured hosts appears in both captures; it is kept once
  (`twws.extract.deduplicate`, per connection, so that both directions come from the same
  capture: among connections with the same protocol, originator and responder address and
  port, scanned in order of start time, a record that starts within 1 s of the last retained
  one and comes from a capture that has not yet contributed an observation of it is removed;
  a record from a capture that has already contributed one is a new connection that reuses
  the five-tuple and is kept).

Snort replays the same captures
(`snort -c <conf> -r <capture> -A fast -l <dir> -K none -U -y -k none`; its standard output and
error are kept per capture as `snort.stdout` and `snort.stderr`, and a non-zero exit code stops
the extraction); an alert is mapped
to every Zeek connection with the same transport protocol and the same two endpoints (address
and port, in either order) whose lifetime, extended by 1 s on both sides, contains the time of
the alert; alerts without port numbers cannot be mapped. With `snort.rules: []` the rule files
enabled in the distribution's `snort.conf`
are used; their names, rule counts and SHA-256 digests are written to
`extract/extract_info.json` and to the supplementary material.

Zeek and Snort tasks of one day run in one pool of worker processes, largest capture first
(`extract.n_jobs`; `null` = number of cores minus one).

Which alerts count for the signature oracle is decided by Snort's own classification of its rules
(the `classtype` of each rule): only alerts of text rules (generator 1) whose class is listed in
`snort.attack_classtypes` (21 attack classes, see `configs/paper.yaml`) set the flag
`snort_alert` of a connection. Informational, policy and protocol-decode classes and the events of
preprocessors and decoders (which carry no class) do not count. The initial version of the oracle
counted every alert; that flag is kept as `snort_alert_any`, and the oracle table and the
ablations of the paper report both. The alert counts per class are written to
`corpus/stream_info.json` (`snort_alerts_by_class`) and to the supplementary material.

### Ground-truth labels (evaluation only)

Labels are derived from Table 2 of the dataset page (transcribed in
`configs/attack_schedule_ids2018.yaml`), not from the published CSV files:

* a flow is **malicious** if one of its endpoints is an attacker address of that capture day,
  at any time of the day; its family is the family of that day's attack whose time window is
  closest (every capture day of the schedule carries attacks of one family);
* the dataset page states attack times without a time zone, so the UTC offset is **estimated**:
  the whole-hour offset that puts the largest share of attacker flows inside the scheduled
  windows, each widened by 5 minutes on both sides; ties are resolved in favour of the
  schedule's default offset (`twws.labels.estimate_utc_offset`; the diagnostic table is written
  to the supplementary material). The time windows are used for this estimate and for the rule
  on ambiguous flows only;
* on the two infiltration days, connections that a compromised victim opens to other internal
  hosts during the attack window are **Ambiguous** (second-stage activity mixed with the host's
  normal traffic): they are excluded from the corpus but stay in the host streams from which
  temporal features are computed;
* the dataset page names Heartbleed among the scenarios but its schedule has no Heartbleed
  entry, so the corpus has **no Heartbleed class**.

### Evaluation corpus

From the stream of all retained flows (`<work_dir>/corpus/stream.parquet`):

1. Ambiguous flows are never selected.
2. A family with at most `corpus.family_cap` (100,000) flows is kept whole; a larger one is
   reduced to `family_cap` flows drawn uniformly at random.
3. Benign flows are drawn uniformly at random so that they make up `corpus.benign_share` (83 %).
4. If more than `corpus.max_rows` flows remain, every class is thinned by the same factor.

Seed: `corpus.sample_seed = 42`. The counts before and after are written to
`corpus/corpus_info.json` and appear in the corpus table of the paper.

## 2. CICIDS2017 (second corpus)

Dataset page: <https://www.unb.ca/cic/datasets/ids-2017.html>. Five daily captures
(Monday to Friday, 3–7 July 2017), offered together with an MD5 file each. Put them in one folder
and run the three commands at the top of `configs/ids2017.yaml`. Labels follow
`configs/attack_schedule_ids2017.yaml`:

* a flow is malicious if one of its endpoints is an attacker address, at any time of the day;
* the captures show most attack traffic after the firewall's address translation, so a flow between
  the firewall's internal address (`nat_address`) and a victim of an attack is malicious when it lies
  inside that attack's window (widened by `window_slack_seconds`);
* several days carry attacks of different families by the same attacker. A flow involved in several
  attacks of a day is named after the attack whose victim it reaches, and otherwise (or among those)
  after the attack with the nearest window. Without the victim rule, the bots of the Friday capture,
  which contact the attacker throughout the afternoon, would be named after the port scan
  (`twws.labels.FAMILY_RULE`, recorded in `corpus/stream_info.json`);
* the connections that the compromised host opens to internal addresses during the infiltration
  window are Ambiguous, as for CSE-CIC-IDS2018.

The clock offset is estimated as for CSE-CIC-IDS2018. Because the bots contact the attacker outside the
scheduled botnet window as well, a smaller share of the flows with an attacker address falls inside
the windows than for CSE-CIC-IDS2018, and the estimation prints a warning; the binary labels of the
NAT rule are not affected by it. The corpus is used

* as the **labelled source** of the transfer oracle (`twws.cli transfer`), and
* as the **target** of the cross-dataset experiment, where the detector is trained on
  CSE-CIC-IDS2018 without the transfer oracle, so that no CICIDS2017 label is used anywhere.

## 3. Synthetic flows

`twws/synth.py` generates flow tables with the schema of the Zeek extraction. They exist to
test the code. Nothing derived from them is reported.
