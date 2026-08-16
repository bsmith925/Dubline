from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_faq_covers_core_user_journey_and_product_boundaries():
    page = (ROOT / "web" / "faq.html").read_text(encoding="utf-8")
    assert page.count('class="faq-item"') == 24
    for required in (
        "What is Dubline?", "Needs review", "voice rights", "local-path option",
        "does not author or reconstruct Atmos objects", "does not download network URLs",
        "not a guarantee of biometric identity", "not full-film generative face replacement",
    ):
        assert required in page


def test_main_screen_links_to_faq_and_faq_assets_exist():
    index = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    assert 'href="/faq"' in index
    assert (ROOT / "web" / "faq.css").is_file()
    assert (ROOT / "web" / "faq.js").is_file()
