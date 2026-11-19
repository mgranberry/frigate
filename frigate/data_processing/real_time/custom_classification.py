"""Real time processor for custom classification models."""

import datetime
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any

import cv2
import numpy as np

from frigate.comms.embeddings_updater import EmbeddingsRequestEnum
from frigate.comms.event_metadata_updater import EventMetadataPublisher
from frigate.comms.inter_process import InterProcessRequestor
from frigate.config import FrigateConfig
from frigate.config.classification import CustomClassificationConfig
from frigate.const import CLIPS_DIR, MODEL_CACHE_DIR
from frigate.detectors.detector_config import (
    InputDTypeEnum,
    InputTensorEnum,
    PixelFormatEnum,
)
from frigate.log import suppress_stderr_during
from frigate.object_detection.util import tensor_transform
from frigate.util.builtin import EventsPerSecond, InferenceSpeed, load_labels
from frigate.util.classification import (
    CLASSIFICATION_INPUT_SIZE,
    resize_classification_image,
)
from frigate.util.file import trim_oldest_files
from frigate.util.image import calculate_region
from frigate.util.model import get_ort_providers
from frigate.util.object import box_overlaps

from ..types import DataProcessorMetrics
from .api import DeferredRealtimeProcessorApi

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ModuleNotFoundError:
    from ai_edge_litert.interpreter import Interpreter, load_delegate

logger = logging.getLogger(__name__)

MAX_OBJECT_CLASSIFICATIONS = 16


def _crop_and_resize_object(
    rgb: np.ndarray,
    box: list[int] | tuple[int, int, int, int],
    target_size: tuple[int, int],
    strategy: str = "square",
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop an object from an RGB frame and resize according to strategy."""
    xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    img_h, img_w = rgb.shape[:2]
    if strategy == "square":
        x, y, x2, y2 = calculate_region(
            rgb.shape,
            xmin,
            ymin,
            xmax,
            ymax,
            int(max(xmax - xmin, ymax - ymin) * scale),
            1.0,
        )
        crop = rgb[y:y2, x:x2]
        return crop, resize_classification_image(crop, target_size)

    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    half_w = (xmax - xmin) * scale / 2.0
    half_h = (ymax - ymin) * scale / 2.0
    x = max(0, int(cx - half_w))
    y = max(0, int(cy - half_h))
    x2 = min(img_w, max(x + 1, int(cx + half_w)))
    y2 = min(img_h, max(y + 1, int(cy + half_h)))
    crop = rgb[y:y2, x:x2]

    if strategy == "letterbox":
        tw, th = target_size
        ch, cw = crop.shape[:2]
        ratio = min(tw / max(1, cw), th / max(1, ch))
        nw = max(1, int(round(cw * ratio)))
        nh = max(1, int(round(ch * ratio)))
        resized = cv2.resize(crop, (nw, nh))
        canvas = np.zeros((th, tw, 3), dtype=np.uint8)
        y_off = (th - nh) // 2
        x_off = (tw - nw) // 2
        canvas[y_off : y_off + nh, x_off : x_off + nw] = resized
        return crop, canvas

    return crop, resize_classification_image(crop, target_size)


def _normalize_scores(raw: np.ndarray) -> np.ndarray:
    """Convert raw logits or unnormalized scores into a 1D float32 probability vector."""
    arr = np.asarray(raw, dtype=np.float32).ravel()
    if arr.size == 0:
        return arr
    total = float(arr.sum())
    if (
        np.all(arr >= 0.0)
        and total > 0.0
        and (abs(total - 1.0) <= 1e-2 or np.max(arr) <= 1.0)
    ):
        return arr / total
    exps = np.exp(arr - np.max(arr))
    return np.asarray(exps / exps.sum(), dtype=np.float32)


class ClassificationBackend(ABC):
    """Runtime backend interface for custom classification models."""

    backend_name: str
    input_size: tuple[int, int]

    def __init__(
        self,
        width: int = CLASSIFICATION_INPUT_SIZE[0],
        height: int = CLASSIFICATION_INPUT_SIZE[1],
        input_tensor: InputTensorEnum = InputTensorEnum.nhwc,
        input_pixel_format: PixelFormatEnum = PixelFormatEnum.rgb,
        input_dtype: InputDTypeEnum = InputDTypeEnum.int,
    ) -> None:
        self.input_size = (int(width), int(height))
        self.input_transform = tensor_transform(input_tensor)
        self.input_pixel_format = input_pixel_format
        self.input_dtype = input_dtype

    def _transform_input(self, resized_rgb: np.ndarray) -> np.ndarray:
        """Format an RGB (H, W, 3) uint8 crop to match the model's declared input spec."""
        img = (
            cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)
            if self.input_pixel_format == PixelFormatEnum.bgr
            else resized_rgb
        )
        tensor_input: np.ndarray = np.expand_dims(img, axis=0)
        if self.input_transform:
            tensor_input = np.transpose(tensor_input, self.input_transform)
        if self.input_dtype == InputDTypeEnum.float:
            tensor_input = tensor_input.astype(np.float32) / 255.0
        elif self.input_dtype == InputDTypeEnum.float_denorm:
            tensor_input = tensor_input.astype(np.float32)
        return tensor_input

    @abstractmethod
    def predict(self, resized_rgb: np.ndarray) -> np.ndarray:
        """Run inference on an RGB crop sized to input_size and return 1D probabilities."""


class TFLiteClassifierBackend(ClassificationBackend):
    """TFLite / LiteRT backend supporting CPU and Coral EdgeTPU delegates."""

    def __init__(
        self,
        model_path: str,
        device: str | None = None,
        width: int = CLASSIFICATION_INPUT_SIZE[0],
        height: int = CLASSIFICATION_INPUT_SIZE[1],
        input_tensor: InputTensorEnum = InputTensorEnum.nhwc,
        input_pixel_format: PixelFormatEnum = PixelFormatEnum.rgb,
        input_dtype: InputDTypeEnum = InputDTypeEnum.int,
    ) -> None:
        super().__init__(
            width=width,
            height=height,
            input_tensor=input_tensor,
            input_pixel_format=input_pixel_format,
            input_dtype=input_dtype,
        )
        delegates = []
        dev = (device or "").strip()
        if dev.lower() in ("edgetpu", "usb", "pci") or dev.lower().startswith(
            ("usb:", "pci:")
        ):
            delegate_opts = {} if dev.lower() == "edgetpu" else {"device": dev}
            delegates = [load_delegate("libedgetpu.so.1.0", delegate_opts)]
        self.backend_name = "EdgeTPU" if delegates else "TFLite"
        with suppress_stderr_during("tflite_interpreter_init"):
            self.interpreter = Interpreter(
                model_path=model_path,
                num_threads=2,
                experimental_delegates=delegates or None,
            )
            self.interpreter.allocate_tensors()
        self.tensor_input_details = self.interpreter.get_input_details()
        self.tensor_output_details = self.interpreter.get_output_details()

    def predict(self, resized_rgb: np.ndarray) -> np.ndarray:
        tensor_in = self._transform_input(resized_rgb)
        self.interpreter.set_tensor(self.tensor_input_details[0]["index"], tensor_in)
        self.interpreter.invoke()
        res: np.ndarray = self.interpreter.get_tensor(
            self.tensor_output_details[0]["index"]
        )[0]
        return _normalize_scores(res)


class ONNXClassifierBackend(ClassificationBackend):
    """ONNX Runtime backend using Frigate's get_ort_providers()."""

    def __init__(
        self,
        model_path: str,
        device: str | None = "AUTO",
        width: int = CLASSIFICATION_INPUT_SIZE[0],
        height: int = CLASSIFICATION_INPUT_SIZE[1],
        input_tensor: InputTensorEnum = InputTensorEnum.nhwc,
        input_pixel_format: PixelFormatEnum = PixelFormatEnum.rgb,
        input_dtype: InputDTypeEnum = InputDTypeEnum.int,
    ) -> None:
        super().__init__(
            width=width,
            height=height,
            input_tensor=input_tensor,
            input_pixel_format=input_pixel_format,
            input_dtype=input_dtype,
        )
        import onnxruntime as ort

        dev = device or "AUTO"
        providers, provider_options = get_ort_providers(
            force_cpu=(dev == "CPU"), device=dev
        )
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=providers,
            provider_options=provider_options,
        )
        self.backend_name = session.get_providers()[0]
        self.input_name = session.get_inputs()[0].name

        in_w, in_h = self.input_size
        warm_arr = self._transform_input(np.zeros((in_h, in_w, 3), dtype=np.uint8))
        for _ in range(2):
            session.run(None, {self.input_name: warm_arr})
        self.session = session

    def predict(self, resized_rgb: np.ndarray) -> np.ndarray:
        tensor_in = self._transform_input(resized_rgb)
        res = self.session.run(None, {self.input_name: tensor_in})[0][0]
        return _normalize_scores(res)


class RKNNClassifierBackend(ClassificationBackend):
    """Rockchip NPU backend using Frigate's RKNNModelRunner for .rknn models."""

    backend_name = "RKNN"

    def __init__(
        self,
        model_path: str,
        width: int = CLASSIFICATION_INPUT_SIZE[0],
        height: int = CLASSIFICATION_INPUT_SIZE[1],
        input_tensor: InputTensorEnum = InputTensorEnum.nhwc,
        input_pixel_format: PixelFormatEnum = PixelFormatEnum.rgb,
        input_dtype: InputDTypeEnum = InputDTypeEnum.int,
    ) -> None:
        super().__init__(
            width=width,
            height=height,
            input_tensor=input_tensor,
            input_pixel_format=input_pixel_format,
            input_dtype=input_dtype,
        )
        from frigate.detectors.detection_runners import RKNNModelRunner

        self.runner = RKNNModelRunner(model_path=model_path)
        self.input_name = self.runner.get_input_names()[0]

    def predict(self, resized_rgb: np.ndarray) -> np.ndarray:
        tensor_in = self._transform_input(resized_rgb)
        outputs = self.runner.run({self.input_name: tensor_in})
        return _normalize_scores(outputs[0][0])


def load_classification_backend(
    model_dir: str,
    model_config: CustomClassificationConfig,
) -> tuple[ClassificationBackend | None, dict[int, str]]:
    """Load the first available classification backend (.rknn, .onnx, .tflite) and labelmap."""
    labelmap_path = os.path.join(model_dir, "labelmap.txt")
    if not os.path.exists(labelmap_path):
        return None, {}

    shape_kwargs: dict[str, Any] = {
        "width": model_config.width,
        "height": model_config.height,
        "input_tensor": model_config.input_tensor,
        "input_pixel_format": model_config.input_pixel_format,
        "input_dtype": model_config.input_dtype,
    }
    common_kwargs: dict[str, Any] = {"device": model_config.device, **shape_kwargs}

    rknn_path = os.path.join(model_dir, "model.rknn")
    if os.path.exists(rknn_path):
        return (
            RKNNClassifierBackend(rknn_path, **shape_kwargs),
            load_labels(labelmap_path, prefill=0, indexed=False),
        )

    onnx_path = os.path.join(model_dir, "model.onnx")
    if os.path.exists(onnx_path):
        if (model_config.device or "").upper() == "RKNN":
            from frigate.util.rknn_converter import auto_convert_model

            converted = auto_convert_model(onnx_path)
            if converted:
                return (
                    RKNNClassifierBackend(converted, **shape_kwargs),
                    load_labels(labelmap_path, prefill=0, indexed=False),
                )
        return (
            ONNXClassifierBackend(onnx_path, **common_kwargs),
            load_labels(labelmap_path, prefill=0, indexed=False),
        )

    tflite_path = os.path.join(model_dir, "model.tflite")
    if os.path.exists(tflite_path):
        return (
            TFLiteClassifierBackend(tflite_path, **common_kwargs),
            load_labels(labelmap_path, prefill=0, indexed=False),
        )

    return None, {}


class CustomStateClassificationProcessor(DeferredRealtimeProcessorApi):
    def __init__(
        self,
        config: FrigateConfig,
        model_config: CustomClassificationConfig,
        requestor: InterProcessRequestor,
        metrics: DataProcessorMetrics,
    ):
        super().__init__(config, metrics, max_queue=4)
        self.model_config = model_config

        if not self.model_config.name:
            raise ValueError("Custom classification model name must be set.")

        self.requestor = requestor
        self.model_dir = os.path.join(MODEL_CACHE_DIR, self.model_config.name)
        self.train_dir = os.path.join(CLIPS_DIR, self.model_config.name, "train")
        self.backend: ClassificationBackend | None = None
        self.labelmap: dict[int, str] = {}
        self.classifications_per_second = EventsPerSecond()
        self.state_history: dict[str, dict[str, Any]] = {}

        if (
            self.metrics
            and self.model_config.name in self.metrics.classification_speeds
        ):
            self.inference_speed: InferenceSpeed | None = InferenceSpeed(
                self.metrics.classification_speeds[self.model_config.name]
            )
        else:
            self.inference_speed = None

        # monotonic so a backward wall clock step can't pause classification
        self.last_run = time.monotonic()
        self.__build_detector()

    def __build_detector(self) -> None:
        self.backend, self.labelmap = load_classification_backend(
            self.model_dir, self.model_config
        )
        if self.backend is not None:
            self._forget_unknown_states()
            self.classifications_per_second.start()

    def _forget_unknown_states(self) -> None:
        """Drop verified states that are not labels of the loaded model.

        A retrained model can rename or remove labels. Keeping a state it can
        no longer produce would report its first verified state as a change
        from that obsolete label.
        """
        labels = set(self.labelmap.values())
        self.state_history = {
            camera: history
            for camera, history in self.state_history.items()
            if history["current_state"] in labels
        }

    def __update_metrics(self, duration: float) -> None:
        self.classifications_per_second.update()
        if self.inference_speed:
            self.inference_speed.update(duration)

    def _should_save_image(
        self, camera: str, detected_state: str, score: float = 1.0
    ) -> bool:
        """
        Determine if we should save the image for training.
        Save when:
        - State is changing or being verified (regardless of score)
        - Score is less than 100% (even if state matches, useful for training)
        Don't save when:
        - State is stable (matches current_state) AND score is 100%
        """
        if camera not in self.state_history:
            # First detection for this camera, save it
            return True

        verification = self.state_history[camera]
        current_state = verification.get("current_state")
        pending_state = verification.get("pending_state")

        # Save if there's a pending state change being verified
        if pending_state is not None:
            return True

        # Save if the detected state differs from the current verified state
        # (state is changing)
        if current_state is not None and detected_state != current_state:
            return True

        # If score is less than 100%, save even if state matches
        # (useful for training to improve confidence)
        if score < 1.0:
            return True

        # Don't save if state is stable (detected_state == current_state) AND score is 100%
        return False

    def verify_state_change(
        self, camera: str, detected_state: str, timestamp: float
    ) -> tuple[str | None, float] | None:
        """
        Verify state change requires 3 consecutive identical states before publishing.
        Returns (previous state, time the new state was first seen) once verified,
        or None if verification not complete. The previous state is None for the
        first state verified on a camera.
        """
        if camera not in self.state_history:
            self.state_history[camera] = {
                "current_state": None,
                "pending_state": None,
                "pending_since": 0.0,
                "consecutive_count": 0,
            }

        verification = self.state_history[camera]

        if detected_state == verification["current_state"]:
            verification["pending_state"] = None
            verification["consecutive_count"] = 0
            return None

        if detected_state == verification["pending_state"]:
            verification["consecutive_count"] += 1

            if verification["consecutive_count"] >= 3:
                previous_state = verification["current_state"]
                verification["current_state"] = detected_state
                verification["pending_state"] = None
                verification["consecutive_count"] = 0
                return previous_state, verification["pending_since"]
        else:
            verification["pending_state"] = detected_state
            verification["pending_since"] = timestamp
            verification["consecutive_count"] = 1
            logger.debug(
                f"New state '{detected_state}' detected for {camera}, need {3 - verification['consecutive_count']} more consecutive detections"
            )

        return None

    def process_frame(self, frame_data: dict[str, Any], frame: np.ndarray) -> None:
        if not self.model_config.name or not self.model_config.state_config:
            return

        if self.metrics and self.model_config.name in self.metrics.classification_cps:
            self.metrics.classification_cps[
                self.model_config.name
            ].value = self.classifications_per_second.eps()
        camera = str(frame_data.get("camera"))

        if camera not in self.model_config.state_config.cameras:
            return

        camera_config = self.model_config.state_config.cameras[camera]
        crop = [
            camera_config.crop[0] * self.config.cameras[camera].detect.width,
            camera_config.crop[1] * self.config.cameras[camera].detect.height,
            camera_config.crop[2] * self.config.cameras[camera].detect.width,
            camera_config.crop[3] * self.config.cameras[camera].detect.height,
        ]
        should_run = False

        now = datetime.datetime.now().timestamp()
        mono_now = time.monotonic()
        if (
            self.model_config.state_config.interval
            and mono_now > self.last_run + self.model_config.state_config.interval
        ):
            self.last_run = mono_now
            should_run = True

        if (
            not should_run
            and self.model_config.state_config.motion
            and any([box_overlaps(crop, mb) for mb in frame_data.get("motion", [])])
        ):
            # classification should run at most once per second
            if mono_now > self.last_run + 1:
                self.last_run = mono_now
                should_run = True

        # Shortcut: always run if we have a pending state verification to complete
        if (
            not should_run
            and camera in self.state_history
            and self.state_history[camera]["pending_state"] is not None
            and mono_now > self.last_run + 0.5
        ):
            self.last_run = mono_now
            should_run = True
            logger.debug(
                f"Running verification check for pending state: {self.state_history[camera]['pending_state']} ({self.state_history[camera]['consecutive_count']}/3)"
            )

        if not should_run:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_I420)
        height, width = rgb.shape[:2]

        # Convert normalized crop coordinates to pixel values
        x1 = int(camera_config.crop[0] * width)
        y1 = int(camera_config.crop[1] * height)
        x2 = int(camera_config.crop[2] * width)
        y2 = int(camera_config.crop[3] * height)

        # Clip coordinates to frame boundaries
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            logger.warning(
                f"Invalid crop coordinates for {camera}: [{x1}, {y1}, {x2}, {y2}]"
            )
            return

        cropped_frame = rgb[y1:y2, x1:x2]

        try:
            target_size = (
                self.backend.input_size
                if self.backend is not None
                else CLASSIFICATION_INPUT_SIZE
            )
            resized_frame = resize_classification_image(cropped_frame, target_size)
        except Exception:
            logger.warning("Failed to resize image for state classification")
            return

        # Copy for training image saves on worker thread
        crop_bgr = cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2BGR)

        self._enqueue_task(("classify", camera, now, resized_frame, crop_bgr))

    def _process_task(self, task: Any) -> None:
        kind = task[0]
        if kind == "classify":
            _, camera, timestamp, resized_frame, crop_bgr = task
            self._classify_state(camera, timestamp, resized_frame, crop_bgr)
        elif kind == "reload":
            self.__build_detector()

    def _classify_state(
        self,
        camera: str,
        timestamp: float,
        resized_frame: np.ndarray,
        crop_bgr: np.ndarray,
    ) -> None:
        if self.backend is None:
            # When backend is None, always save (score is 0.0, which is < 1.0)
            if self._should_save_image(camera, "unknown", 0.0):
                save_attempts = (
                    self.model_config.save_attempts
                    if self.model_config.save_attempts is not None
                    else 100
                )
                write_classification_attempt(
                    self.train_dir,
                    crop_bgr,
                    "none-none",
                    timestamp,
                    "unknown",
                    0.0,
                    max_files=save_attempts,
                )
            return

        probs = self.backend.predict(resized_frame)
        logger.debug(
            f"{self.model_config.name} Ran state classification with probabilities: {probs}"
        )
        best_id = int(np.argmax(probs))
        score = round(probs[best_id], 2)
        self.__update_metrics(datetime.datetime.now().timestamp() - timestamp)

        detected_state = self.labelmap[best_id]

        if self._should_save_image(camera, detected_state, score):
            save_attempts = (
                self.model_config.save_attempts
                if self.model_config.save_attempts is not None
                else 100
            )
            write_classification_attempt(
                self.train_dir,
                crop_bgr,
                "none-none",
                timestamp,
                detected_state,
                score,
                max_files=save_attempts,
            )

        if score < self.model_config.threshold:
            logger.debug(
                f"Score {score} below threshold {self.model_config.threshold}, skipping verification"
            )
            return

        verified = self.verify_state_change(camera, detected_state, timestamp)

        if verified is not None:
            previous_state, changed_at = verified
            self._emit_result(
                {
                    "type": "classification",
                    "processor": "state",
                    "model_name": self.model_config.name,
                    "camera": camera,
                    "state": detected_state,
                    "previous_state": previous_state,
                    "timestamp": changed_at,
                }
            )

    def handle_request(
        self, topic: str, request_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        if topic == EmbeddingsRequestEnum.reload_classification_model.value:
            if request_data.get("model_name") == self.model_config.name:

                def _do_reload(data: dict[str, Any]) -> dict[str, Any]:
                    self.__build_detector()
                    logger.info(
                        f"Successfully loaded updated model for {self.model_config.name}"
                    )
                    return {
                        "success": True,
                        "message": f"Loaded {self.model_config.name} model.",
                    }

                result: dict[str, Any] = self._enqueue_request(_do_reload, request_data)
                return result
            else:
                return None
        else:
            return None

    def expire_object(self, object_id: str, camera: str) -> None:
        pass


class CustomObjectClassificationProcessor(DeferredRealtimeProcessorApi):
    def __init__(
        self,
        config: FrigateConfig,
        model_config: CustomClassificationConfig,
        sub_label_publisher: EventMetadataPublisher,
        requestor: InterProcessRequestor,
        metrics: DataProcessorMetrics,
    ):
        super().__init__(config, metrics, max_queue=8)
        self.model_config = model_config

        if not self.model_config.name:
            raise ValueError("Custom classification model name must be set.")

        self.model_dir = os.path.join(MODEL_CACHE_DIR, self.model_config.name)
        self.train_dir = os.path.join(CLIPS_DIR, self.model_config.name, "train")
        self.backend: ClassificationBackend | None = None
        self.sub_label_publisher = sub_label_publisher
        self.requestor = requestor
        self.classification_history: dict[str, list[tuple[str, float, float]]] = {}
        self.labelmap: dict[int, str] = {}
        self.classifications_per_second = EventsPerSecond()

        if (
            self.metrics
            and self.model_config.name in self.metrics.classification_speeds
        ):
            self.inference_speed: InferenceSpeed | None = InferenceSpeed(
                self.metrics.classification_speeds[self.model_config.name]
            )
        else:
            self.inference_speed = None

        self.__build_detector()

    def __build_detector(self) -> None:
        self.backend, self.labelmap = load_classification_backend(
            self.model_dir, self.model_config
        )
        if self.backend is not None:
            logger.info(
                "Loaded %s object classifier (%dx%d) for %s",
                self.backend.backend_name,
                self.backend.input_size[0],
                self.backend.input_size[1],
                self.model_config.name,
            )

    def __update_metrics(self, duration: float) -> None:
        self.classifications_per_second.update()
        if self.inference_speed:
            self.inference_speed.update(duration)

    def get_weighted_score(
        self,
        object_id: str,
        current_label: str,
        current_score: float,
        current_time: float,
    ) -> tuple[str | None, float]:
        """
        Determine weighted score based on history to prevent false positives/negatives.
        Requires 60% of attempts to agree on a label before publishing.
        Returns (weighted_label, weighted_score) or (None, 0.0) if no weighted score.
        """
        if object_id not in self.classification_history:
            self.classification_history[object_id] = []
            logger.debug(f"Created new classification history for {object_id}")

        self.classification_history[object_id].append(
            (current_label, current_score, current_time)
        )

        history = self.classification_history[object_id]
        logger.debug(
            f"History for {object_id}: {len(history)} entries, latest=({current_label}, {current_score})"
        )

        if len(history) < 3:
            logger.debug(
                f"History for {object_id} has {len(history)} entries, need at least 3"
            )
            return None, 0.0

        label_counts: dict[str, int] = {}
        label_scores: dict[str, list[float]] = {}
        total_attempts = len(history)

        for label, score, timestamp in history:
            if label not in label_counts:
                label_counts[label] = 0
                label_scores[label] = []

            label_counts[label] += 1
            label_scores[label].append(score)

        best_label = max(label_counts, key=lambda k: label_counts[k])
        best_count = label_counts[best_label]

        consensus_threshold = total_attempts * 0.6
        logger.debug(
            f"Consensus calc for {object_id}: label_counts={label_counts}, "
            f"best_label={best_label}, best_count={best_count}, "
            f"total={total_attempts}, threshold={consensus_threshold}"
        )

        if best_count < consensus_threshold:
            logger.debug(
                f"No consensus for {object_id}: {best_count} < {consensus_threshold}"
            )
            return None, 0.0

        avg_score = sum(label_scores[best_label]) / len(label_scores[best_label])

        if best_label == "none":
            logger.debug(f"Filtering 'none' label for {object_id}")
            return None, 0.0

        logger.debug(
            f"Consensus reached for {object_id}: {best_label} with avg_score={avg_score}"
        )
        return best_label, avg_score

    def process_frame(self, obj_data: dict[str, Any], frame: np.ndarray) -> None:
        if not self.model_config.name or not self.model_config.object_config:
            return

        if self.metrics and self.model_config.name in self.metrics.classification_cps:
            self.metrics.classification_cps[
                self.model_config.name
            ].value = self.classifications_per_second.eps()

        if obj_data["false_positive"]:
            return

        if obj_data["label"] not in self.model_config.object_config.objects:
            return

        if obj_data.get("end_time") is not None:
            return

        object_id = obj_data["id"]

        if (
            object_id in self.classification_history
            and len(self.classification_history[object_id])
            >= MAX_OBJECT_CLASSIFICATIONS
        ):
            return

        now = datetime.datetime.now().timestamp()
        rgb = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_I420)

        try:
            target_size = (
                self.backend.input_size
                if self.backend is not None
                else CLASSIFICATION_INPUT_SIZE
            )
            crop, resized_crop = _crop_and_resize_object(
                rgb,
                obj_data["box"],
                target_size,
                strategy=self.model_config.object_config.crop_strategy,
                scale=self.model_config.object_config.crop_scale,
            )
        except Exception:
            logger.warning("Failed to resize image for object classification")
            return

        # Copy crop for training images (will be used on worker thread)
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

        self._enqueue_task(
            ("classify", object_id, obj_data["camera"], now, resized_crop, crop_bgr)
        )

    def _process_task(self, task: Any) -> None:
        kind = task[0]
        if kind == "classify":
            _, object_id, camera, timestamp, resized_crop, crop_bgr = task
            self._classify_object(object_id, camera, timestamp, resized_crop, crop_bgr)
        elif kind == "expire":
            _, object_id = task
            if object_id in self.classification_history:
                self.classification_history.pop(object_id)
        elif kind == "reload":
            self.__build_detector()

    def _classify_object(
        self,
        object_id: str,
        camera: str,
        timestamp: float,
        resized_crop: np.ndarray,
        crop_bgr: np.ndarray,
    ) -> None:
        if self.backend is None:
            save_attempts = (
                self.model_config.save_attempts
                if self.model_config.save_attempts is not None
                else 200
            )
            write_classification_attempt(
                self.train_dir,
                crop_bgr,
                object_id,
                timestamp,
                "unknown",
                0.0,
                max_files=save_attempts,
            )

            # Still track history even when model doesn't exist to respect MAX_OBJECT_CLASSIFICATIONS
            # Add an entry with "unknown" label so the history limit is enforced
            if object_id not in self.classification_history:
                self.classification_history[object_id] = []

            self.classification_history[object_id].append(("unknown", 0.0, timestamp))
            return

        probs = self.backend.predict(resized_crop)
        logger.debug(
            f"{self.model_config.name} Ran object classification with probabilities: {probs}"
        )
        best_id = int(np.argmax(probs))
        score = round(probs[best_id], 2)
        self.__update_metrics(datetime.datetime.now().timestamp() - timestamp)

        save_attempts = (
            self.model_config.save_attempts
            if self.model_config.save_attempts is not None
            else 200
        )
        write_classification_attempt(
            self.train_dir,
            crop_bgr,
            object_id,
            timestamp,
            self.labelmap[best_id],
            score,
            max_files=save_attempts,
        )

        if score < self.model_config.threshold:
            logger.debug(
                f"{self.model_config.name}: Score {score} < threshold {self.model_config.threshold} for {object_id}, skipping"
            )
            return

        sub_label = self.labelmap[best_id]

        logger.debug(
            f"{self.model_config.name}: Object {object_id} passed threshold with sub_label={sub_label}, score={score}"
        )

        consensus_label, consensus_score = self.get_weighted_score(
            object_id, sub_label, score, timestamp
        )

        logger.debug(
            f"{self.model_config.name}: get_weighted_score returned consensus_label={consensus_label}, consensus_score={consensus_score} for {object_id}"
        )

        if consensus_label is not None and self.model_config.object_config is not None:
            self._emit_result(
                {
                    "type": "classification",
                    "processor": "object",
                    "model_name": self.model_config.name,
                    "classification_type": self.model_config.object_config.classification_type,
                    "object_id": object_id,
                    "camera": camera,
                    "timestamp": timestamp,
                    "label": consensus_label,
                    "score": consensus_score,
                }
            )

    def handle_request(
        self, topic: str, request_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        if topic == EmbeddingsRequestEnum.reload_classification_model.value:
            if request_data.get("model_name") == self.model_config.name:

                def _do_reload(data: dict[str, Any]) -> dict[str, Any]:
                    self.__build_detector()
                    logger.info(
                        f"Successfully loaded updated model for {self.model_config.name}"
                    )
                    return {
                        "success": True,
                        "message": f"Loaded {self.model_config.name} model.",
                    }

                result: dict[str, Any] = self._enqueue_request(_do_reload, request_data)
                return result
            else:
                return None
        else:
            return None

    def expire_object(self, object_id: str, camera: str) -> None:
        self._enqueue_task(("expire", object_id))


def write_classification_attempt(
    folder: str,
    frame: np.ndarray,
    event_id: str,
    timestamp: float,
    label: str,
    score: float,
    max_files: int = 100,
) -> None:
    if "-" in label:
        label = label.replace("-", "_")

    file = os.path.join(folder, f"{event_id}-{timestamp}-{label}-{score}.webp")
    os.makedirs(folder, exist_ok=True)
    cv2.imwrite(file, frame)
    trim_oldest_files(folder, max_files)
