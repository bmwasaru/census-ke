from census_sources import CENSUS_SOURCES, SOURCE_BY_FILENAME


def test_all_four_main_2019_census_volumes_are_configured():
    assert [source["volume"] for source in CENSUS_SOURCES] == ["I", "II", "III", "IV"]


def test_source_filenames_are_unique_and_indexed():
    filenames = [source["filename"] for source in CENSUS_SOURCES]

    assert len(filenames) == len(set(filenames))
    assert set(SOURCE_BY_FILENAME) == set(filenames)


def test_each_source_has_primary_and_fallback_knbs_urls():
    for source in CENSUS_SOURCES:
        assert source["repository_url"].startswith("https://repository.knbs.or.ke/")
        assert source["fallback_url"].startswith("https://www.knbs.or.ke/")
