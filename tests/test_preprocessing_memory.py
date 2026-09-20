"""SAM mask fidelity, OOM recovery, and preprocessing model lifetimes."""

import tempfile
import types
import unittest
import weakref
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import polars as pl
import torch
from segment_anything import SamPredictor
from segment_anything.modeling import MaskDecoder, PromptEncoder, Sam, TwoWayTransformer

from src.preprocessing.steps.depth import DepthModel, DepthStep
from src.preprocessing.steps.segment import SegmentModel, SegmentStep
from src.preprocessing.steps.yolo import YoloModel, YoloStep
from tests.test_preprocessing import _coords_row


def small_predictor() -> SamPredictor:
    """Use real SAM prompts/decoder/postprocessing without a large checkpoint."""
    encoder = torch.nn.Conv2d(3, 32, kernel_size=16, stride=16)
    encoder.img_size = 64
    return SamPredictor(
        Sam(
            image_encoder=encoder,
            prompt_encoder=PromptEncoder(
                embed_dim=32,
                image_embedding_size=(4, 4),
                input_image_size=(64, 64),
                mask_in_chans=16,
            ),
            mask_decoder=MaskDecoder(
                transformer_dim=32,
                transformer=TwoWayTransformer(
                    depth=1, embedding_dim=32, num_heads=4, mlp_dim=64
                ),
            ),
        ).eval()
    )


class SegmentMemoryTests(unittest.TestCase):
    def test_masks_match_sam_for_original_sizes_and_point_coordinates(self):
        model = SegmentModel.__new__(SegmentModel)
        model.model_initialized = True
        model.model = small_predictor()
        rng = np.random.default_rng(42)
        for height, width in ((37, 91), (91, 37), (64, 64)):
            with self.subTest(shape=(height, width)):
                image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
                points = np.array(
                    [[width * 0.8, height * 0.4], [width * 0.2, height * 0.6]]
                )
                labels = np.ones(2)
                model.model.set_image(image)
                expected, _, _ = model.model.predict(
                    point_coords=points, point_labels=labels, multimask_output=False
                )
                with mock.patch("torch.cuda.is_available", return_value=False):
                    actual = model.get_mask(image, points, labels)
                np.testing.assert_array_equal(actual, expected[0])
                self.assertEqual(actual.shape, (height, width))
                self.assertEqual(actual.dtype, np.bool_)
                self.assertFalse(model.model.is_image_set)
                self.assertIsNone(model.model.features)

    def test_cuda_oom_retries_on_cpu_and_next_image_tries_gpu(self):
        model = SegmentModel.__new__(SegmentModel)
        model.model_initialized = True
        model.model = mock.Mock()
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        mask = np.ones((20, 30), dtype=bool)
        devices = []

        def predict(image, points, labels, device):
            devices.append(device.type)
            if len(devices) == 1:
                raise torch.cuda.OutOfMemoryError("simulated encoder OOM")
            return mask

        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.empty_cache") as empty_cache,
            mock.patch.object(model, "_predict_mask", side_effect=predict),
        ):
            self.assertIs(model.get_mask(image, np.zeros((2, 2)), np.ones(2)), mask)
            self.assertIs(model.get_mask(image, np.zeros((2, 2)), np.ones(2)), mask)
        self.assertEqual(devices, ["cuda", "cpu", "cuda"])
        model.model.model.to.assert_called_once_with(torch.device("cpu"))
        empty_cache.assert_called_once()
        self.assertEqual(model.model.reset_image.call_count, 3)

    def test_non_oom_errors_are_not_retried(self):
        model = SegmentModel.__new__(SegmentModel)
        model.model_initialized = True
        model.model = mock.Mock()
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch.object(
                model, "_predict_mask", side_effect=ValueError("bad image")
            ) as predict,
        ):
            with self.assertRaisesRegex(ValueError, "bad image"):
                model.get_mask(np.zeros((10, 10, 3)), np.zeros((2, 2)), np.ones(2))
        predict.assert_called_once()
        model.model.reset_image.assert_called_once()

    def test_cpu_retry_failure_propagates_and_clears_embedding(self):
        model = SegmentModel.__new__(SegmentModel)
        model.model_initialized = True
        model.model = mock.Mock()
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.empty_cache"),
            mock.patch.object(
                model,
                "_predict_mask",
                side_effect=[
                    torch.cuda.OutOfMemoryError("OOM"),
                    RuntimeError("CPU failed"),
                ],
            ) as predict,
        ):
            with self.assertRaisesRegex(RuntimeError, "CPU failed"):
                model.get_mask(np.zeros((10, 10, 3)), np.zeros((2, 2)), np.ones(2))
        self.assertEqual(predict.call_count, 2)
        self.assertEqual(model.model.reset_image.call_count, 2)

    def test_recovered_image_is_retained_and_mask_is_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "raw").mkdir()
            checkpoint = root / "sam.pth"
            checkpoint.write_bytes(b"checkpoint identity")
            cv2.imwrite(str(root / "raw/fish.png"), np.zeros((100, 100, 3), np.uint8))
            config = types.SimpleNamespace(
                dataset=types.SimpleNamespace(
                    output_dir=root / "processed", input_dir=root / "raw", rotate=False
                ),
                model_path=types.SimpleNamespace(sam=checkpoint),
            )
            step = SegmentStep(config)
            step.segment_model.model_initialized = True
            step.segment_model.model = mock.Mock()
            mask = np.zeros((100, 100), dtype=bool)
            mask[40:60, 10:90] = True
            frame = pl.DataFrame([_coords_row("fish.png")])
            with (
                mock.patch("torch.cuda.is_available", return_value=True),
                mock.patch("torch.cuda.empty_cache"),
                mock.patch.object(
                    step.segment_model,
                    "_predict_mask",
                    side_effect=[torch.cuda.OutOfMemoryError("OOM"), mask],
                ) as predict,
                mock.patch.object(
                    step.segment_model,
                    "_get_segmentation_model",
                    side_effect=AssertionError("cached mask should avoid loading SAM"),
                ),
            ):
                first = step.process(frame)
                cached = step.process(frame)
            self.assertEqual(first["name"].to_list(), ["fish.png"])
            self.assertTrue(first.equals(cached))
            self.assertEqual(predict.call_count, 2)
            np.testing.assert_array_equal(
                np.load(root / "processed/segment/fish.png.npy"), mask
            )


class ModelLifetimeTests(unittest.TestCase):
    def test_stages_release_models_on_success_and_failure(self):
        for step_type, model_type, attribute in (
            (YoloStep, YoloModel, "yolo_model"),
            (DepthStep, DepthModel, "depth_model"),
            (SegmentStep, SegmentModel, "segment_model"),
        ):
            for fail in (False, True):
                with self.subTest(stage=step_type.__name__, fail=fail):
                    owner = model_type.__new__(model_type)
                    owner.model_initialized = True
                    owner.model = torch.nn.Linear(1, 1)
                    reference = weakref.ref(owner.model)
                    step = step_type.__new__(step_type)
                    setattr(step, attribute, owner)
                    frame = pl.DataFrame({"name": ["fish.png"]})
                    with (
                        mock.patch("torch.cuda.is_available", return_value=False),
                        mock.patch.object(
                            step,
                            "_process_images",
                            return_value=frame,
                            side_effect=RuntimeError("stage failed") if fail else None,
                        ),
                    ):
                        if fail:
                            with self.assertRaisesRegex(RuntimeError, "stage failed"):
                                step.process(frame)
                        else:
                            self.assertIs(step.process(frame), frame)
                    self.assertFalse(owner.model_initialized)
                    self.assertIsNone(reference())

    def test_release_is_safe_before_lazy_loading(self):
        for model_type in (YoloModel, DepthModel, SegmentModel):
            with self.subTest(model=model_type.__name__):
                owner = model_type.__new__(model_type)
                owner.model_initialized = False
                owner.release()
                owner.release()


if __name__ == "__main__":
    unittest.main()
