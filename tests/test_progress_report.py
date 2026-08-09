"""Tests for the Turkish technical progress report builder."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "Thesis" / "scripts" / "build_progress_report_tr.py"
TEMPLATE_PATH = ROOT / "Thesis" / "reports" / "technical_progress_tr" / "report.md"
REFERENCE_BUILDER_PATH = (
    ROOT / "Thesis" / "scripts" / "make_progress_report_reference_docx.py"
)


def load_builder():
    if not BUILDER_PATH.exists():
        return None
    spec = importlib.util.spec_from_file_location(
        "build_progress_report_tr", BUILDER_PATH
    )
    if spec is None or spec.loader is None:
        return None
    scripts_path = str(BUILDER_PATH.parent)
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_reference_builder():
    if not REFERENCE_BUILDER_PATH.exists():
        return None
    spec = importlib.util.spec_from_file_location(
        "make_progress_report_reference_docx", REFERENCE_BUILDER_PATH
    )
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ReportFactTests(unittest.TestCase):
    """Catch stale row counts, split counts, or headline-result wiring."""

    def setUp(self) -> None:
        self.builder = load_builder()
        if self.builder is None:
            self.fail("Turkish progress-report builder does not exist")

    def test_current_dataset_counts_are_loaded_from_pipeline_artifacts(self):
        facts = self.builder.load_report_facts(ROOT)

        self.assertEqual(
            (
                facts.datasets["data-inside"].raw,
                facts.datasets["data-inside"].processed,
                facts.datasets["data-inside"].test,
            ),
            (302, 282, 42),
        )
        self.assertEqual(
            (
                facts.datasets["data-inside-zoom"].raw,
                facts.datasets["data-inside-zoom"].processed,
                facts.datasets["data-inside-zoom"].test,
            ),
            (906, 839, 128),
        )
        self.assertEqual(
            (
                facts.datasets["data-outside"].raw,
                facts.datasets["data-outside"].processed,
                facts.datasets["data-outside"].test,
            ),
            (1284, 1027, 152),
        )

    def test_headline_results_match_the_verified_result_table(self):
        facts = self.builder.load_report_facts(ROOT)

        self.assertAlmostEqual(facts.results["Controlled"].mae, 0.151263, places=6)
        self.assertAlmostEqual(facts.results["Zoom-derived"].mae, 0.806484, places=6)
        self.assertAlmostEqual(facts.results["Uncontrolled"].mae, 6.149671, places=6)
        self.assertEqual(facts.results["Controlled"].experiment, "linear_coords_depth")
        self.assertEqual(facts.results["Zoom-derived"].experiment, "cnn_eye")
        self.assertEqual(
            facts.results["Uncontrolled"].experiment, "dino_ridge_coords_depth"
        )

    def test_loaded_facts_obey_pipeline_handoff_invariants(self):
        facts = self.builder.load_report_facts(ROOT)

        self.builder.validate_report_facts(facts)
        for dataset in facts.datasets.values():
            self.assertEqual(dataset.raw, dataset.split)
            self.assertEqual(dataset.processed, dataset.predicted)
            self.assertEqual(
                dataset.train + dataset.val + dataset.test, dataset.predicted
            )


class ReportTemplateTests(unittest.TestCase):
    """Catch incomplete, stale, or privacy-unsafe rendered report content."""

    def setUp(self) -> None:
        self.builder = load_builder()
        if self.builder is None:
            self.fail("Turkish progress-report builder does not exist")
        if not TEMPLATE_PATH.is_file():
            self.fail("Turkish progress-report template does not exist")
        if not hasattr(self.builder, "render_report_markdown"):
            self.fail("Report template renderer does not exist")
        self.rendered = self.builder.render_report_markdown(
            TEMPLATE_PATH,
            self.builder.load_report_facts(ROOT),
            Path("/tmp/fishometry-system-diagram.png"),
        )

    def test_rendered_report_contains_the_approved_technical_sections(self):
        required = [
            "İçindekiler",
            "Yönetici Özeti",
            "Genel Sistem Mimarisi",
            "Veri Kümeleri",
            "Görüntü Ön İşleme Pipeline'ı",
            "Eğitim ve Modelleme Pipeline'ı",
            "Elde Edilen Sonuçlar",
            "Tamamlanan Çalışmalar ve Güncel Durum",
            "Sınırlılıklar, Teknik Riskler ve Sonraki Adımlar",
        ]
        for heading in required:
            with self.subTest(heading=heading):
                self.assertIn(heading, self.rendered)

    def test_rendered_report_is_detailed_and_has_no_unresolved_tokens(self):
        self.assertNotRegex(self.rendered, r"\{\{[A-Z0-9_]+\}\}")
        self.assertGreater(len(self.rendered.split()), 3500)
        self.assertIn("0,151", self.rendered)
        self.assertIn("6,150", self.rendered)

    def test_rendered_report_excludes_unneeded_personal_identifiers(self):
        self.assertNotRegex(self.rendered, r"\b\d{11}\b")
        self.assertNotIn("@gmail.com", self.rendered.lower())

    def test_results_section_starts_on_a_new_page(self):
        page_break = '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'

        self.assertIn(f"{page_break}\n```\n\n# Elde Edilen Sonuçlar", self.rendered)

    def test_headline_table_uses_readable_model_labels(self):
        self.assertIn("Linear + koordinat + derinlik", self.rendered)
        self.assertIn("CNN + göz", self.rendered)
        self.assertIn("DINOv2/Ridge + koordinat + derinlik", self.rendered)
        self.assertNotIn("| linear_coords_depth |", self.rendered)


class ReferenceDocTests(unittest.TestCase):
    """Catch page-geometry, typography, and page-furniture regressions."""

    def setUp(self) -> None:
        self.reference_builder = load_reference_builder()
        if self.reference_builder is None:
            self.fail("Progress-report reference DOCX builder does not exist")

    def test_reference_doc_has_a4_geometry_and_report_typography(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "reference.docx"
            self.reference_builder.build_reference_docx(output)
            with zipfile.ZipFile(output) as archive:
                document = archive.read("word/document.xml").decode("utf-8")
                styles = archive.read("word/styles.xml").decode("utf-8")

        self.assertIn('w:w="11906" w:h="16838"', document)
        self.assertIn('w:left="1417"', document)
        self.assertIn('w:right="1247"', document)
        self.assertIn('w:ascii="Times New Roman"', styles)
        self.assertIn('w:styleId="Title"', styles)
        self.assertIn('w:styleId="Heading1"', styles)
        self.assertIn('w:color w:val="17365D"', styles)

    def test_reference_doc_has_running_header_footer_and_live_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "reference.docx"
            self.reference_builder.build_reference_docx(output)
            with zipfile.ZipFile(output) as archive:
                names = set(archive.namelist())
                header = archive.read("word/header1.xml").decode("utf-8")
                footer = archive.read("word/footer1.xml").decode("utf-8")
                settings = archive.read("word/settings.xml").decode("utf-8")

        self.assertIn("word/header1.xml", names)
        self.assertIn("word/footer1.xml", names)
        self.assertIn("Fishometry | Teknik İlerleme Raporu", header)
        self.assertIn("PAGE", footer)
        self.assertIn("updateFields", settings)


class BuildPipelineTests(unittest.TestCase):
    """Catch missing visual output and malformed document-conversion commands."""

    def setUp(self) -> None:
        self.builder = load_builder()
        if self.builder is None:
            self.fail("Turkish progress-report builder does not exist")
        for name in ("make_system_diagram", "pandoc_command"):
            if not hasattr(self.builder, name):
                self.fail(f"Report build function does not exist: {name}")

    def test_system_diagram_is_a_nonempty_high_resolution_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "pipeline.png"
            result = self.builder.make_system_diagram(output)
            self.assertEqual(result, output)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 20_000)

    def test_pandoc_command_uses_static_contents_numbering_and_reference_styles(self):
        command = self.builder.pandoc_command(
            Path("report.md"),
            Path("reference.docx"),
            Path("report.docx"),
            ROOT,
        )
        self.assertNotIn("--toc", command)
        self.assertFalse(any(arg.startswith("--toc-") for arg in command))
        self.assertIn("--number-sections", command)
        self.assertIn("--reference-doc=reference.docx", command)
        self.assertEqual(command[-2:], ["-o", "report.docx"])

    def test_report_table_formatter_repeats_headers_and_uses_content_widths(self):
        if not hasattr(self.builder, "format_report_tables"):
            self.fail("Report table formatter does not exist")
        source = (
            '<w:tbl><w:tblPr><w:tblW w:w="9360" w:type="dxa"/>'
            '<w:tblLayout w:type="fixed"/></w:tblPr>'
            '<w:tr><w:tc><w:tcPr><w:tcW w:w="4680" w:type="dxa"/></w:tcPr>'
            "<w:p/></w:tc></w:tr></w:tbl>"
        )
        formatted = self.builder.format_report_tables(source)
        self.assertIn('<w:tblLayout w:type="autofit"/>', formatted)
        self.assertIn("<w:tblHeader/>", formatted)
        self.assertIn("<w:cantSplit/>", formatted)
        self.assertIn('<w:tcW w:w="0" w:type="auto"/>', formatted)


if __name__ == "__main__":
    unittest.main()
