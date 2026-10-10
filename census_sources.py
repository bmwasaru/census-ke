CENSUS_SOURCES = [
    {
        "volume": "I",
        "title": "Population by County and Sub-County",
        "filename": "2019-kphc-volume-1.pdf",
        "repository_url": "https://repository.knbs.or.ke/bitstreams/4ec8e85f-c86c-45b0-bb62-6b926e6b5c45/download",
        "fallback_url": "https://www.knbs.or.ke/wp-content/uploads/2023/09/2019-Kenya-population-and-Housing-Census-Volume-1-Population-By-County-And-Sub-County.pdf",
    },
    {
        "volume": "II",
        "title": "Distribution of Population by Administrative Units",
        "filename": "2019-kphc-volume-2.pdf",
        "repository_url": "https://repository.knbs.or.ke/bitstreams/f240c7f0-873a-4951-aa4e-7a839106384a/download",
        "fallback_url": "https://www.knbs.or.ke/wp-content/uploads/2023/09/2019-Kenya-population-and-Housing-Census-Volume-2-Distribution-of-Population-by-Administrative-Units.pdf",
    },
    {
        "volume": "III",
        "title": "Distribution of Population by Age and Sex",
        "filename": "2019-kphc-volume-3.pdf",
        "repository_url": "https://repository.knbs.or.ke/bitstreams/e6efb34a-7918-43fc-8504-2d033acf1a8d/download",
        "fallback_url": "https://www.knbs.or.ke/wp-content/uploads/2023/09/2019-Kenya-population-and-Housing-Census-Volume-3-Distribution-of-Population-by-Age-and-Sex.pdf",
    },
    {
        "volume": "IV",
        "title": "Distribution of Population by Socio-Economic Characteristics",
        "filename": "2019-kphc-volume-4.pdf",
        "repository_url": "https://repository.knbs.or.ke/bitstreams/40bc90d8-177f-44d2-a9e4-7b99beb70ec5/download",
        "fallback_url": "https://www.knbs.or.ke/wp-content/uploads/2023/09/2019-Kenya-population-and-Housing-Census-Volume-4-Distribution-of-Population-by-Socio-Economic-Characteristics.pdf",
    },
]
SOURCE_BY_FILENAME = {item["filename"]: item for item in CENSUS_SOURCES}
