from urllib.error import URLError

import download_census


def sample_source():
    return {
        "volume": "I",
        "title": "Population by County and Sub-County",
        "filename": "volume-1.pdf",
        "repository_url": "https://repository.knbs.or.ke/primary.pdf",
        "fallback_url": "https://www.knbs.or.ke/fallback.pdf",
    }


def test_download_source_skips_existing_large_file(tmp_path, monkeypatch):
    source = sample_source()
    destination = tmp_path / source["filename"]
    destination.write_bytes(b"x" * 60_000)
    monkeypatch.setattr(download_census, "DATA_DIR", tmp_path)

    def should_not_download(*args, **kwargs):
        raise AssertionError("download should not have been called")

    monkeypatch.setattr(download_census, "download", should_not_download)

    assert download_census.download_source(source) is True


def test_download_source_falls_back_to_second_url(tmp_path, monkeypatch):
    source = sample_source()
    monkeypatch.setattr(download_census, "DATA_DIR", tmp_path)
    calls = []

    def fake_download(url, destination):
        calls.append(url)
        if "repository" in url:
            raise URLError("primary unavailable")
        destination.write_bytes(b"p" * 60_000)

    monkeypatch.setattr(download_census, "download", fake_download)

    assert download_census.download_source(source) is True
    assert calls == [source["repository_url"], source["fallback_url"]]
    assert (tmp_path / source["filename"]).stat().st_size == 60_000


def test_download_source_returns_false_if_all_urls_fail(tmp_path, monkeypatch):
    source = sample_source()
    monkeypatch.setattr(download_census, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        download_census,
        "download",
        lambda *args, **kwargs: (_ for _ in ()).throw(URLError("offline")),
    )

    assert download_census.download_source(source) is False
    assert not (tmp_path / source["filename"]).exists()
