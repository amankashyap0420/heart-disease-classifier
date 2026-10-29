from streamlit.testing.v1 import AppTest

from heart.config import ROOT


def test_app_loads_and_classifies_example():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert not app.error
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert any("Predicted dataset class" in element.value for element in app.subheader)
    assert any(element.label == "Model score for class 1" for element in app.metric)


def test_unknown_category_is_handled():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    app.selectbox(key="thal").set_value("Unknown")
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert any("imputed" in element.value for element in app.warning)


def test_all_unknown_record_is_rejected():
    from heart.config import CATEGORIES, NUMERIC
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    for name in CATEGORIES:
        app.selectbox(key=name).set_value("Unknown")
    for name in NUMERIC:
        app.checkbox(key=f"missing_{name}").check()
    app.button[0].click().run()
    assert not app.exception
    assert any("all features missing" in element.value for element in app.error)
