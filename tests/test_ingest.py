import pymupdf as fitz

from ingest import (
    chunk_text,
    clean_cell,
    clean_text,
    normalize_headers,
    prose_outside_tables,
    rects_intersect,
    rows_from_table_data,
    table_number_from_title,
    table_row_to_search_text,
)


def test_clean_cell_removes_extra_whitespace():

    assert clean_cell("  MOMBASA \n County  ") == "MOMBASA County"


def test_clean_cell_handles_none():

    assert clean_cell(None) == ""


def test_clean_text_joins_hyphenated_words():

    text = "popula-\n" "tion statistics"

    cleaned = clean_text(text)

    assert "population statistics" in cleaned


def test_chunk_text_creates_overlap():

    text = " ".join(f"word{i}" for i in range(20))

    chunks = chunk_text(
        text,
        chunk_size=10,
        overlap=2,
    )

    assert len(chunks) == 3

    first_words = chunks[0].split()

    second_words = chunks[1].split()

    assert first_words[-2:] == second_words[:2]


def test_normalize_headers_handles_duplicates():

    headers = normalize_headers(
        [
            "Male",
            "Male",
            "",
        ]
    )

    assert headers == [
        "Male",
        "Male 2",
        "Column 3",
    ]


def test_rows_from_table_data():

    table = [
        [
            "County/Sub County",
            "Households Farming",
            "Maize",
            "Beans",
        ],
        [
            "MOMBASA",
            "12,497",
            "5,681",
            "2,849",
        ],
        [
            "KISAUNI",
            "6,257",
            "3,456",
            "1,702",
        ],
    ]

    columns, rows = rows_from_table_data(table)

    assert columns == [
        "County/Sub County",
        "Households Farming",
        "Maize",
        "Beans",
    ]

    assert len(rows) == 2

    assert rows[0]["County/Sub County"] == "MOMBASA"

    assert rows[0]["Maize"] == "5,681"


def test_rows_from_table_data_with_external_headers():

    table = [
        [
            "MOMBASA",
            "12,497",
            "5,681",
        ]
    ]

    columns, rows = rows_from_table_data(
        table,
        header_names=[
            "Area",
            "Households",
            "Maize",
        ],
    )

    assert columns == [
        "Area",
        "Households",
        "Maize",
    ]

    assert rows[0]["Area"] == "MOMBASA"


def test_rows_ignore_empty_rows():

    table = [
        [
            "Area",
            "Maize",
        ],
        [
            "",
            "",
        ],
        [
            "MOMBASA",
            "5,681",
        ],
    ]

    _, rows = rows_from_table_data(table)

    assert len(rows) == 1

    assert rows[0]["Area"] == "MOMBASA"


def test_table_number_from_title():

    title = "Table 2.22: Distribution " "of Households Growing Crops"

    assert table_number_from_title(title) == "2.22"


def test_table_number_missing():

    assert table_number_from_title("Population statistics") is None


def test_table_row_to_search_text():

    row = {
        "County/Sub County": "MOMBASA",
        "Households Farming": "12,497",
        "Maize": "5,681",
        "Rice": "-",
        "Beans": "2,849",
    }

    text = table_row_to_search_text(
        ("Table 2.22: " "Distribution of " "Households Growing Crops"),
        row,
    )

    assert "MOMBASA" in text

    assert "Maize: 5,681" in text

    assert "Beans: 2,849" in text

    # Dash-only values should not pollute
    # semantic-search text.
    assert "Rice:" not in text


def test_rectangles_intersect():

    first = fitz.Rect(
        0,
        0,
        100,
        100,
    )

    second = fitz.Rect(
        50,
        50,
        150,
        150,
    )

    assert (
        rects_intersect(
            first,
            second,
        )
        is True
    )


def test_rectangles_do_not_intersect():

    first = fitz.Rect(
        0,
        0,
        100,
        100,
    )

    second = fitz.Rect(
        200,
        200,
        300,
        300,
    )

    assert (
        rects_intersect(
            first,
            second,
        )
        is False
    )


class FakePage:

    def get_text(
        self,
        mode,
    ):
        assert mode == "blocks"

        return [
            (
                0,
                0,
                500,
                50,
                "Normal paragraph",
                0,
                0,
            ),
            (
                0,
                100,
                500,
                300,
                "Ugly flattened table",
                0,
                0,
            ),
        ]


def test_prose_outside_tables():

    page = FakePage()

    table_bbox = (
        0,
        100,
        500,
        300,
    )

    text = prose_outside_tables(
        page,
        [table_bbox],
    )

    assert "Normal paragraph" in text

    assert "Ugly flattened table" not in text
