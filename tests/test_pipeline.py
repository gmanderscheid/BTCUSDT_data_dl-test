"""
Tests UNITAIRES du pipeline et des contrôles, sur données synthétiques.

Pourquoi : les tests de données (test_data_quality.py) appliquent des contrôles aux vrais
fichiers. Encore faut-il que ces contrôles détectent ce qu'ils prétendent détecter, et que
le pipeline produise ce qu'il prétend produire. Ici, chaque cas a un résultat connu à
l'avance :
- la reconstruction des bougies 1 s est comparée à des valeurs calculées à la main, avec
  plusieurs tailles de morceaux, avec et sans en-tête, en ms et en µs, avec doublons ;
- le téléchargement est testé contre un faux serveur (checksum juste, faux, absent, vide,
  erreur 500) ;
- la fusion, le remplacement (--force) et la réparation sont testés sur des fichiers
  temporaires ;
- chaque contrôle de qualité reçoit des cas POSITIFS (défaut légitime qui doit passer,
  par exemple des trades déplacés d'une minute) et NÉGATIFS (défaut réel qui doit
  échouer, dont les contre-exemples de la revue du 26/09/2026).

Ces tests n'ont besoin d'aucune donnée réelle : `pytest -m unit` s'exécute en quelques
secondes.
"""
from __future__ import annotations

import hashlib
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

import download_binance as db
import quality_checks as qc
from gaps import dataset_gaps, find_gaps, merge_adjacent_gaps

pytestmark = pytest.mark.unit

T0 = pd.Timestamp("2024-03-10 12:00:00", tz="UTC")
T0_MS = int(T0.value // 10**6)
SPEC_1S = db.DATASETS["futures_klines_1s"]


# =========================================================================== outils

def trades_frame(rows):
    return pd.DataFrame(rows, columns=db.AGG_TRADE_COLS)


def write_zip(path: Path, df: pd.DataFrame, header: bool = True) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("x.csv", df.to_csv(index=False, header=header))
    return path


# 5 aggTrades sur 2 secondes, résultat calculé À LA MAIN :
#   seconde 0 : open 100, high 101, low 99, close 99, volume 4, quote 401,
#               n_trades 4, acheteur 2 BTC / 199 USDT
#   seconde 1 : open 102, high 102, low 98, close 98, volume 4, quote 404,
#               n_trades 3, acheteur 1 BTC / 98 USDT
BASE_TRADES = [
    (1, 100.0, 1.0, 1, 1, T0_MS + 100, "false"),
    (2, 101.0, 2.0, 2, 3, T0_MS + 500, "true"),
    (3, 99.0, 1.0, 4, 4, T0_MS + 999, "false"),
    (4, 102.0, 3.0, 5, 5, T0_MS + 1000, "true"),
    (5, 98.0, 1.0, 6, 7, T0_MS + 1999, "false"),
]
EXPECTED_BARS = pd.DataFrame({
    "open_time": [T0, T0 + pd.Timedelta(seconds=1)],
    "open": [100.0, 102.0], "high": [101.0, 102.0], "low": [99.0, 98.0], "close": [99.0, 98.0],
    "volume": [4.0, 4.0], "quote_volume": [401.0, 404.0], "n_trades": [4, 3],
    "taker_buy_base": [2.0, 1.0], "taker_buy_quote": [199.0, 98.0],
})


def check_bars(bars: pd.DataFrame) -> None:
    cols = list(EXPECTED_BARS.columns)
    got = bars[cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(got, EXPECTED_BARS, check_dtype=False, check_index_type=False)
    assert (bars["close_time"] - bars["open_time"] == pd.Timedelta(milliseconds=999)).all()


# =========================================================================== reconstruction 1 s

class TestReconstruction:

    @pytest.mark.parametrize("chunk", [1, 2, 3, 100])
    @pytest.mark.parametrize("header", [True, False])
    def test_exact_bars(self, tmp_path, monkeypatch, chunk, header):
        """OHLCV, n_trades et volume acheteur exacts, quelle que soit la taille des morceaux."""
        monkeypatch.setattr(db, "CHUNK_ROWS", chunk)
        z = write_zip(tmp_path / "a.zip", trades_frame(BASE_TRADES), header=header)
        check_bars(db.build_bars_from_aggtrades(z, SPEC_1S))

    def test_capitalized_booleans_and_microseconds(self, tmp_path):
        """is_buyer_maker en « True/False » et horodatages en microsecondes."""
        rows = [(i, p, q, f, l, t * 1000, m.capitalize()) for i, p, q, f, l, t, m in BASE_TRADES]
        z = write_zip(tmp_path / "a.zip", trades_frame(rows))
        check_bars(db.build_bars_from_aggtrades(z, SPEC_1S))

    @pytest.mark.parametrize("chunk", [2, 100])
    def test_adjacent_duplicates_removed(self, tmp_path, monkeypatch, chunk):
        """Chaque ligne en double : résultat identique à l'original."""
        monkeypatch.setattr(db, "CHUNK_ROWS", chunk)
        dup = [r for r in BASE_TRADES for _ in (0, 1)]
        z = write_zip(tmp_path / "a.zip", trades_frame(dup))
        check_bars(db.build_bars_from_aggtrades(z, SPEC_1S))

    @pytest.mark.parametrize("chunk", [2, 100])
    def test_repeated_block_removed(self, tmp_path, monkeypatch, chunk):
        """Un bloc répété à la fin du fichier (cas de septembre 2022) : résultat identique."""
        monkeypatch.setattr(db, "CHUNK_ROWS", chunk)
        z = write_zip(tmp_path / "a.zip", trades_frame(BASE_TRADES + BASE_TRADES[1:4]))
        check_bars(db.build_bars_from_aggtrades(z, SPEC_1S))

    @pytest.mark.parametrize("chunk", [2, 10])
    def test_unordered_new_id_fails_loudly(self, tmp_path, monkeypatch, chunk):
        """
        Identifiants 1, 3, 2, 4 (contre-exemple de la revue) : l'identifiant 2 est inédit mais
        hors ordre. Il ne doit JAMAIS être supprimé silencieusement : la reconstruction échoue,
        quelle que soit la taille des morceaux.
        """
        monkeypatch.setattr(db, "CHUNK_ROWS", chunk)
        rows = [BASE_TRADES[0], BASE_TRADES[2], BASE_TRADES[1], BASE_TRADES[3]]
        rows = [(i, p, q, f, l, T0_MS + k, m) for k, (i, p, q, f, l, t, m) in enumerate(rows)]
        z = write_zip(tmp_path / "a.zip", trades_frame(rows))
        with pytest.raises(db.AggTradeOrderError):
            db.build_bars_from_aggtrades(z, SPEC_1S)

    def test_id_ranges(self):
        r = db.IdRanges()
        r.add(np.array([1, 2, 3, 7, 8]))
        r.add(np.array([4, 5]))
        assert r.contains(np.array([1, 5, 6, 7, 9])).tolist() == [True, True, False, True, False]
        assert len(r.starts) == 2   # [1-5] et [7-8] fusionnés


# =========================================================================== téléchargement

class FakeResponse:
    def __init__(self, status: int, body: bytes = b""):
        self.status_code, self.body = status, body
        self.text = body.decode()

    @property
    def ok(self):
        return self.status_code < 400

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size):
        yield self.body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeSession:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, **kw):
        self.calls.append(url)
        return self.routes.get(url, FakeResponse(404))


ZIP_BODY = b"contenu du zip"
GOOD_SHA = hashlib.sha256(ZIP_BODY).hexdigest()
URL = f"{db.BASE}/x/a.zip"


@pytest.fixture
def fake_net(monkeypatch):
    monkeypatch.setattr(db.time, "sleep", lambda s: None)

    def install(checksum: FakeResponse | None, zip_status: int = 200):
        routes = {URL: FakeResponse(zip_status, ZIP_BODY)}
        if checksum is not None:
            routes[URL + ".CHECKSUM"] = checksum
        session = FakeSession(routes)
        monkeypatch.setattr(db, "session", session)
        return session
    return install


class TestDownload:

    def test_good_checksum(self, tmp_path, fake_net):
        fake_net(FakeResponse(200, f"{GOOD_SHA}  a.zip".encode()))
        assert db.download("x/a.zip", tmp_path / "a.zip") == GOOD_SHA
        assert (tmp_path / "a.zip").read_bytes() == ZIP_BODY

    def test_wrong_checksum_fails_and_cleans(self, tmp_path, fake_net):
        fake_net(FakeResponse(200, f"{'0' * 64}  a.zip".encode()))
        with pytest.raises(db.ChecksumError):
            db.download("x/a.zip", tmp_path / "a.zip")
        assert not (tmp_path / "a.zip").exists()

    @pytest.mark.parametrize("body", [b"", b"pas-un-sha  a.zip"])
    def test_empty_or_malformed_checksum_fails(self, tmp_path, fake_net, body):
        fake_net(FakeResponse(200, body))
        with pytest.raises(db.ChecksumError):
            db.download("x/a.zip", tmp_path / "a.zip")

    def test_checksum_server_error_is_retried_then_fails(self, tmp_path, fake_net):
        """Réponse 500 pour le CHECKSUM (contre-exemple de la revue) : le fichier n'est PAS accepté."""
        session = fake_net(FakeResponse(500))
        with pytest.raises(requests.HTTPError):
            db.download("x/a.zip", tmp_path / "a.zip")
        assert session.calls.count(URL + ".CHECKSUM") == db.RETRIES
        assert not (tmp_path / "a.zip").exists()

    def test_missing_checksum_blocks_unless_allowed(self, tmp_path, fake_net, monkeypatch):
        fake_net(None)
        with pytest.raises(db.ChecksumUnavailable):
            db.download("x/a.zip", tmp_path / "a.zip")
        monkeypatch.setattr(db, "ALLOW_MISSING_CHECKSUM", True)
        assert db.download("x/a.zip", tmp_path / "a.zip") == GOOD_SHA

    def test_missing_zip_returns_none(self, tmp_path, fake_net):
        fake_net(None, zip_status=404)
        assert db.download("x/a.zip", tmp_path / "a.zip") is None


# =========================================================================== stockage et réparation

def bars(start: pd.Timestamp, n: int, close: float = 100.0, freq="1s") -> pd.DataFrame:
    t = pd.date_range(start, periods=n, freq=freq)
    return pd.DataFrame({"open_time": t, "close": close, "volume": 1.0})


class TestStorage:

    def test_merge_is_idempotent_and_add_only(self, tmp_path):
        p = tmp_path / "f.parquet"
        assert db.merge_into(p, bars(T0, 5), "open_time") == 5
        assert db.merge_into(p, bars(T0, 5, close=999.0), "open_time") == 0
        assert (pd.read_parquet(p)["close"] == 100.0).all()   # fusion : l'existant n'est pas modifié

    def test_replace_range_replaces_values(self, tmp_path):
        p = tmp_path / "f.parquet"
        db.merge_into(p, bars(T0, 10), "open_time")
        start, end = T0 + pd.Timedelta(seconds=2), T0 + pd.Timedelta(seconds=5)
        db.replace_range(p, bars(T0, 10, close=101.0), "open_time", start, end)
        df = pd.read_parquet(p)
        assert len(df) == 10 and not df["open_time"].duplicated().any()
        inside = (df["open_time"] >= start) & (df["open_time"] < end)
        assert (df.loc[inside, "close"] == 101.0).all() and (df.loc[~inside, "close"] == 100.0).all()

    def _sync_env(self, tmp_path, monkeypatch, served: pd.DataFrame):
        """Faux téléchargement : chaque ZIP demandé contient `served`."""
        monkeypatch.setattr(db, "DATA_DIR", tmp_path / "raw")
        monkeypatch.setattr(db, "DOWNLOAD_DIR", tmp_path / "dl")
        monkeypatch.setattr(db, "MANIFEST", tmp_path / "manifest.csv")

        def fake_download(rel, dest):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"x")
            return "sha"
        monkeypatch.setattr(db, "download", fake_download)
        return {**SPEC_1S, "builder": lambda path, spec: served, "daily": None}

    def test_force_replaces_existing_month(self, tmp_path, monkeypatch):
        """--force (contre-exemple de la revue) : la valeur corrigée REMPLACE l'ancienne."""
        month = date(2024, 3, 1)
        out = tmp_path / "raw" / "ds" / "ds_2024-03.parquet"
        spec = self._sync_env(tmp_path, monkeypatch, bars(pd.Timestamp(month, tz="UTC"), 5, close=101.0))
        db.merge_into(out, bars(pd.Timestamp(month, tz="UTC"), 5, close=100.0), "open_time")
        monkeypatch.setattr(db, "START", month)
        monkeypatch.setattr(db, "month_starts", lambda s, e: iter([month]))
        db.sync_dataset("ds", spec, force=True, repair=False)
        assert (pd.read_parquet(out)["close"] == 101.0).all()
        assert pd.read_csv(tmp_path / "manifest.csv")["mode"].tolist() == ["replace"]

    @pytest.mark.parametrize("served_n, expected_close", [(10, 101.0), (3, 100.0)])
    def test_replace_only_if_superset(self, tmp_path, monkeypatch, served_n, expected_close):
        """
        Réparation d'une journée : remplacée si le fichier journalier couvre tout l'existant
        (10 secondes servies pour 5 présentes), sinon simple fusion (3 servies).
        """
        out = tmp_path / "raw" / "ds" / "f.parquet"
        spec = self._sync_env(tmp_path, monkeypatch, bars(T0, served_n, close=101.0))
        db.merge_into(out, bars(T0, 5, close=100.0), "open_time")
        db.fetch_and_store("x/a.zip", out, spec, "test", mode="replace_if_superset", period=db.day_bounds(T0.date()))
        df = pd.read_parquet(out)
        assert (df.loc[df["open_time"] == T0, "close"] == expected_close).all()

    def test_day_with_all_minutes_but_missing_volume_is_flagged(self, tmp_path):
        """Journée amputée en contenu, sans minute disparue : signalée pour réparation."""
        day0 = pd.Timestamp("2024-03-01", tz="UTC")
        t = pd.date_range(day0, periods=2 * 1440 * 60, freq="1s")
        s = pd.DataFrame({"open_time": t, "volume": 1.0})
        s.loc[s["open_time"] >= day0 + pd.Timedelta(days=1), "volume"] = 0.7   # 2e jour : -30 %
        m = pd.DataFrame({"open_time": pd.date_range(day0, periods=2 * 1440, freq="1min"), "volume": 60.0})
        s.to_parquet(tmp_path / "s.parquet")
        m.to_parquet(tmp_path / "m.parquet")
        todo = db.perp_day_discrepancies(tmp_path / "s.parquet", tmp_path / "m.parquet")
        assert list(todo["futures_klines_1s"]) == [date(2024, 3, 2)]


# =========================================================================== trous

class TestGaps:

    def test_find_gaps(self):
        ts = pd.Series(pd.date_range(T0, periods=10, freq="1s").delete([3, 4, 5]))
        g = find_gaps(ts, T0, T0 + pd.Timedelta(seconds=10))
        assert g["duration_s"].tolist() == [3]

    def test_gap_split_by_month_change_is_merged(self, tmp_path):
        """Un trou de 90 s coupé par minuit (40 s + 50 s) est bien classé comme long."""
        d = tmp_path / "spot_klines_1s"
        d.mkdir()
        end_a = pd.Timestamp("2024-04-01", tz="UTC")
        a = pd.date_range(end_a - pd.Timedelta(hours=1), end_a - pd.Timedelta(seconds=41), freq="1s")
        b = pd.date_range(end_a + pd.Timedelta(seconds=50), end_a + pd.Timedelta(hours=1), freq="1s")
        pd.DataFrame({"open_time": a}).to_parquet(d / "spot_klines_1s_2024-03.parquet")
        pd.DataFrame({"open_time": b}).to_parquet(d / "spot_klines_1s_2024-04.parquet")
        dataset_gaps.cache_clear()
        g = dataset_gaps("spot_klines_1s", str(tmp_path), today=date(2024, 6, 1))
        boundary = g[(g["start"] <= end_a) & (g["end"] >= end_a)]
        assert boundary["duration_s"].tolist() == [90]

    def test_merge_adjacent(self):
        g = pd.DataFrame({"start": [T0, T0 + pd.Timedelta(seconds=40)],
                          "end": [T0 + pd.Timedelta(seconds=39), T0 + pd.Timedelta(seconds=89)],
                          "duration_s": [40, 50]})
        assert merge_adjacent_gaps(g)["duration_s"].tolist() == [90]


# =========================================================================== contrôles des bougies 1 s

def kline_frame(n=5, start=T0):
    t = pd.date_range(start, periods=n, freq="1s")
    return pd.DataFrame({
        "open_time": t, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
        "volume": 2.0, "close_time": t + pd.Timedelta(milliseconds=999), "quote_volume": 200.0,
        "n_trades": np.int64(3), "taker_buy_base": 1.0, "taker_buy_quote": 100.0,
    })


class TestKlineChecks:

    def test_clean_frame_passes(self):
        df = kline_frame()
        assert not qc.duration_errors(df) and not qc.vwap_errors(df, 0.001)
        assert not qc.zero_consistency_errors(df) and not qc.price_jump_errors(df, 0.10)

    def test_negative_duration_before_gap_fails(self):
        """close_time une seconde avant open_time, suivie d'une seconde absente (contre-exemple)."""
        df = kline_frame().drop(index=3).reset_index(drop=True)
        df.loc[2, "close_time"] = df.loc[2, "open_time"] - pd.Timedelta(seconds=1)
        assert any("close_time < open_time" in e for e in qc.duration_errors(df))

    def test_truncated_candle_rules(self):
        df = kline_frame()
        df.loc[4, "close_time"] = df.loc[4, "open_time"] + pd.Timedelta(milliseconds=300)
        assert not qc.duration_errors(df)                                   # fin de fichier : seconde suivante absente
        nxt = df.loc[4, "open_time"] + pd.Timedelta(seconds=1)
        assert qc.duration_errors(df, next_file_first=nxt)                  # présente dans le fichier suivant
        assert not qc.duration_errors(df, next_file_first=nxt + pd.Timedelta(seconds=5))

    def test_microsecond_end_is_normal(self):
        df = kline_frame()
        df["close_time"] = df["open_time"] + pd.Timedelta(microseconds=999_999)
        assert not qc.duration_errors(df)

    def test_infinite_n_trades_fails(self):
        """n_trades = +inf (contre-exemple de la revue)."""
        df = kline_frame()
        df["n_trades"] = df["n_trades"].astype(float)
        df.loc[1, "n_trades"] = np.inf
        assert qc.finite_errors(df, ["n_trades"]) and qc.integer_errors(df, "n_trades")

    @pytest.mark.parametrize("ret, fails", [(-0.095, False), (0.095, False), (-0.105, True), (0.105, True)])
    def test_price_jump_is_symmetric_in_simple_return(self, ret, fails):
        """-9,5 % ne doit pas échouer avec un seuil de 10 % (contre-exemple de la revue)."""
        df = kline_frame()
        df.loc[3:, "close"] = 100.0 * (1 + ret)
        assert bool(qc.price_jump_errors(df, 0.10)) is fails

    def test_buyer_vwap_out_of_range_fails(self):
        df = kline_frame()
        df.loc[2, "taker_buy_quote"] = 150.0            # 150 USDT pour 1 BTC, hors de [99, 101]
        assert any("acheteur" in e for e in qc.vwap_errors(df, 0.001))

    def test_quote_without_base_fails(self):
        df = kline_frame()
        df.loc[1, ["volume", "taker_buy_base"]] = 0.0
        assert qc.zero_consistency_errors(df)


# =========================================================================== contrôles de vérification

def minute_frames(n_days=1, vol=60.0, price=100.0):
    idx = pd.date_range(pd.Timestamp("2024-03-01", tz="UTC"), periods=n_days * 1440, freq="1min")
    off = pd.DataFrame({"open": price, "high": price + 1, "low": price - 1, "close": price,
                        "volume": vol, "quote_volume": vol * price, "taker_buy_base": vol / 2}, index=idx)
    return off.copy(), off


class TestVerificationChecks:

    def test_identical_passes(self):
        ours, off = minute_frames()
        assert not qc.price_errors(ours, off, 0.0005)
        assert not qc.drift_errors(ours, off, 0.001)
        assert not qc.local_volume_errors(ours, off, 0.005, 0.05, 100)

    def test_shifted_trades_are_accepted(self):
        """Cas POSITIF : des trades déplacés dans la minute voisine ne doivent pas échouer."""
        ours, off = minute_frames()
        for i in range(10, 1400, 37):
            ours.iloc[i, ours.columns.get_loc("volume")] += 20
            ours.iloc[i + 1, ours.columns.get_loc("volume")] -= 20
        assert not qc.drift_errors(ours, off, 0.001)
        assert not qc.local_volume_errors(ours, off, 0.005, 0.05, 100)

    def test_flat_bars_with_missing_wicks_fail(self):
        """OHLC plats à 100 contre officiel 99 / 110 / 90 / 101 (contre-exemple de la revue)."""
        ours, off = minute_frames()
        off.iloc[100, off.columns.get_indexer(["open", "high", "low", "close"])] = [99, 110, 90, 101]
        ours[["open", "high", "low", "close"]] = 100.0
        errors = qc.price_errors(ours, off, 0.0005)
        assert any("mèche disparue" in e for e in errors)
        assert qc.close_mismatch_share(ours, off) > 0

    def test_neighbours_are_exact_minutes(self):
        """Voisines 00:00 / 00:01 / 02:00 : 02:00 ne doit pas élargir la référence (contre-exemple)."""
        t = pd.Timestamp("2024-03-01", tz="UTC")
        idx = pd.DatetimeIndex([t, t + pd.Timedelta(minutes=1), t + pd.Timedelta(hours=2)])
        off = pd.DataFrame({"high": [100.0, 100.0, 200.0], "low": 99.0}, index=idx)
        ours = off.copy()
        ours.loc[idx[1], "high"] = 150.0
        assert any("au-dessus" in e for e in qc.price_errors(ours, off, 0.0005))

    def test_one_hour_loss_is_caught_locally(self):
        """
        Une heure amputée de 30 % dans un mois uniforme de 31 jours (contre-exemple) : la
        dérive mensuelle la laisse passer (0,04 % du mois), le contrôle local l'attrape.
        """
        ours, off = minute_frames(n_days=31)
        hour = (ours.index >= ours.index[0] + pd.Timedelta(days=10)) & (ours.index < ours.index[0] + pd.Timedelta(days=10, hours=1))
        ours.loc[hour, ["volume", "taker_buy_base"]] *= 0.7
        assert not qc.drift_errors(ours, off, 0.001)
        assert qc.local_volume_errors(ours, off, 0.005, 0.05, 100)

    def test_buyer_side_inversion_fails(self):
        ours, off = minute_frames()
        off["taker_buy_base"] = off["volume"] * 0.6
        ours["taker_buy_base"] = ours["volume"] * 0.4        # sens acheteur / vendeur inversé
        assert any("taker_buy_base" in e for e in qc.drift_errors(ours, off, 0.001))

    def test_doubled_day_fails(self):
        ours, off = minute_frames(n_days=3)
        ours.loc[ours.index.date == ours.index[0].date(), ["volume", "taker_buy_base"]] *= 2
        assert qc.drift_errors(ours, off, 0.001)
        assert qc.month_volume_gap(ours, off) > 0.01

    def test_nan_drift_fails(self):
        ours, off = minute_frames()
        ours.iloc[5, ours.columns.get_loc("volume")] = np.nan
        assert qc.drift_errors(ours, off, 0.001)


# =========================================================================== registre des minutes

def registry(rows):
    df = pd.DataFrame(rows, columns=["side", "minute", "official_volume", "status", "reason"])
    df["minute"] = pd.to_datetime(df["minute"], utc=True)
    return df


class TestRegistryChecks:

    def test_candidate_blocks_and_approved_excludes(self):
        ours, off = minute_frames()
        gone = off.index[50]
        ours = ours.drop(index=gone)
        start, end = off.index[0], off.index[-1] + pd.Timedelta(minutes=1)
        _, _, errors, _ = qc.apply_registry(ours, off, registry([("absente_chez_nous", gone, 60, "candidate", "")]), start, end)
        assert errors                                           # inscrite mais non examinée
        o2, f2, errors, stats = qc.apply_registry(
            ours, off, registry([("absente_chez_nous", gone, 60, "approved", "vérifié")]), start, end)
        assert not errors and gone not in f2.index
        assert 0 < stats["excluded_official_volume_share"] < 0.001

    def test_stale_exception_fails(self):
        ours, off = minute_frames()
        start, end = off.index[0], off.index[-1] + pd.Timedelta(minutes=1)
        reg = registry([("absente_chez_nous", off.index[50], 60, "approved", "vérifié")])
        _, _, errors, _ = qc.apply_registry(ours, off, reg, start, end)
        assert any("obsolète" in e for e in errors)

    def test_registry_structure(self):
        t = "2024-03-01 00:00:00+00:00"
        assert qc.registry_errors(registry([("autre", t, 0, "approved", "x")]))
        assert qc.registry_errors(registry([("zone_invalide", t, 0, "approved", "")]))
        assert qc.registry_errors(registry([("zone_invalide", t, 0, "approved", "x")] * 2))
        assert qc.registry_errors(registry([("zone_invalide", "2024-03-01 00:00:30+00:00", 0, "approved", "x")]))
        assert not qc.registry_errors(registry([("zone_invalide", t, 0, "approved", "incident")]))


# =========================================================================== funding

def funding_month(times=None):
    start = pd.Timestamp("2024-03-01", tz="UTC")
    t = pd.date_range(start, start + pd.offsets.MonthBegin(1), freq="8h", inclusive="left") if times is None else times
    t = pd.DatetimeIndex(t) + pd.Timedelta(milliseconds=7)       # Binance : quelques ms après l'heure
    return pd.DataFrame({"calc_time": t, "funding_interval_hours": 8, "last_funding_rate": 0.0001})


class TestFundingChecks:
    START = pd.Timestamp("2024-03-01", tz="UTC")
    END = pd.Timestamp("2024-04-01", tz="UTC")

    def test_complete_month_passes(self):
        assert not qc.funding_errors(funding_month(), self.START, self.END)

    @pytest.mark.parametrize("drop", ["first", "last", "internal"])
    def test_missing_payment_fails(self, drop):
        df = funding_month()
        i = {"first": 0, "last": len(df) - 1, "internal": 40}[drop]
        assert qc.funding_errors(df.drop(index=i), self.START, self.END)

    def test_single_payment_fails(self):
        """Un seul paiement au milieu du mois (contre-exemple de la revue)."""
        assert qc.funding_errors(funding_month().iloc[[45]], self.START, self.END)

    def test_extra_payment_fails(self):
        df = funding_month()
        extra = df.iloc[[10]].assign(calc_time=df["calc_time"].iloc[10] + pd.Timedelta(hours=4))
        assert qc.funding_errors(pd.concat([df, extra]).sort_values("calc_time"), self.START, self.END)

    def test_unit_error_is_caught_by_median(self):
        df = funding_month()
        df["last_funding_rate"] = 0.01          # 0,01 % lu comme 1 %
        assert qc.funding_rate_errors(df, 0.03, 0.001)
        assert not qc.funding_rate_errors(funding_month(), 0.03, 0.001)
