"""WMS client for LROC NAC imagery and LOLA topography.

Wraps OGC-standard Web Map Service requests with disk caching,
automatic retries, and helpers for the Lunar Mapping and Modelling
Portal (``wms.im-ldi.com``).
"""

from __future__ import annotations

import hashlib
import io
import time
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import requests
import yaml
from PIL import Image

from moonpiercer.constants import (
    LOLA_DTM_LAYER,
    NAC_OBSERVATION_LAYER,
    NAC_STAMP_LAYER,
    WMS_BASE_URL,
)
from moonpiercer.geometry import normalize_lon


class WMSClient:
    """Cacheable WMS client for the Lunar Mapping and Modelling Portal."""

    def __init__(
        self,
        base_url: str = WMS_BASE_URL,
        cache_dir: Path | str = Path("cache") / "wms",
        use_cache: bool = True,
        timeout_s: int = 60,
        max_retries: int = 4,
    ) -> None:
        self.base_url = base_url
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Low-level request with disk cache + retry
    # ------------------------------------------------------------------

    def _cache_key(self, params: dict) -> str:
        q = urlencode(sorted(params.items()), doseq=True)
        return hashlib.sha1(q.encode("utf-8")).hexdigest()

    def request(self, params: dict, *, binary: bool = True) -> bytes:
        """Issue a WMS request, reading from / writing to disk cache."""
        ext = ".bin" if binary else ".txt"
        cache_path = self.cache_dir / f"{self._cache_key(params)}{ext}"

        if self.use_cache and cache_path.exists():
            return cache_path.read_bytes()

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                r = requests.get(
                    self.base_url, params=params, timeout=self.timeout_s
                )
                r.raise_for_status()
                data = r.content
                if self.use_cache:
                    cache_path.write_bytes(data)
                return data
            except requests.RequestException as exc:
                last_err = exc
                time.sleep(0.6 * (2 ** attempt))

        raise last_err  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Payload helpers
    # ------------------------------------------------------------------

    @staticmethod
    def decode_text(data: bytes) -> str:
        return data.decode("utf-8", errors="ignore")

    @staticmethod
    def is_exception_payload(data: bytes) -> bool:
        return "ServiceExceptionReport" in data[:1500].decode(
            "utf-8", errors="ignore"
        )

    # ------------------------------------------------------------------
    # GetFeatureInfo (metadata queries)
    # ------------------------------------------------------------------

    def get_featureinfo_yaml(
        self,
        layer: str,
        lon_deg: float,
        lat_deg: float,
        search_radius_deg: float = 0.05,
        feature_count: int = 20,
    ) -> list[dict]:
        """Return parsed YAML feature-info records for *layer* near a point."""
        lon_min = normalize_lon(lon_deg - search_radius_deg)
        lon_max = normalize_lon(lon_deg + search_radius_deg)
        if float(lon_min) > float(lon_max):
            return []

        params = {
            "SERVICE": "WMS",
            "VERSION": "1.1.1",
            "REQUEST": "GetFeatureInfo",
            "SRS": "EPSG:4326",
            "BBOX": f"{lon_min},{lat_deg - search_radius_deg},{lon_max},{lat_deg + search_radius_deg}",
            "WIDTH": "512",
            "HEIGHT": "512",
            "LAYERS": layer,
            "QUERY_LAYERS": layer,
            "STYLES": "",
            "FORMAT": "image/png",
            "X": "256",
            "Y": "256",
            "INFO_FORMAT": "text/yaml",
            "FEATURE_COUNT": str(feature_count),
        }
        try:
            data = self.request(params, binary=True)
        except requests.RequestException:
            return []
        if self.is_exception_payload(data):
            return []
        parsed = yaml.safe_load(self.decode_text(data))
        if not isinstance(parsed, dict):
            return []
        values = parsed.get(layer, [])
        return values if isinstance(values, list) else []

    # ------------------------------------------------------------------
    # GetMap (imagery + terrain)
    # ------------------------------------------------------------------

    def get_map_png(
        self,
        layer: str,
        bbox: tuple[float, float, float, float],
        width_px: int,
        height_px: int,
    ) -> Image.Image | None:
        """Fetch a PNG map tile and return as a PIL RGBA Image."""
        lon_min, lat_min, lon_max, lat_max = bbox
        params = {
            "SERVICE": "WMS",
            "VERSION": "1.1.1",
            "REQUEST": "GetMap",
            "SRS": "EPSG:4326",
            "BBOX": f"{lon_min},{lat_min},{lon_max},{lat_max}",
            "WIDTH": str(width_px),
            "HEIGHT": str(height_px),
            "LAYERS": layer,
            "STYLES": "",
            "FORMAT": "image/png",
            "TRANSPARENT": "TRUE",
        }
        try:
            data = self.request(params, binary=True)
        except requests.RequestException:
            return None
        if self.is_exception_payload(data):
            return None
        return Image.open(io.BytesIO(data)).convert("RGBA")

    def get_map_float_tiff(
        self,
        layer: str,
        bbox: tuple[float, float, float, float],
        width_px: int,
        height_px: int,
    ) -> np.ndarray | None:
        """Fetch a 32-bit float TIFF tile and return as a numpy array."""
        lon_min, lat_min, lon_max, lat_max = bbox
        params = {
            "SERVICE": "WMS",
            "VERSION": "1.1.1",
            "REQUEST": "GetMap",
            "SRS": "EPSG:4326",
            "BBOX": f"{lon_min},{lat_min},{lon_max},{lat_max}",
            "WIDTH": str(width_px),
            "HEIGHT": str(height_px),
            "LAYERS": layer,
            "STYLES": "",
            "FORMAT": "image/tiff; mode=32bit",
            "TRANSPARENT": "TRUE",
        }
        try:
            data = self.request(params, binary=True)
        except requests.RequestException:
            return None
        if self.is_exception_payload(data):
            return None
        return np.asarray(Image.open(io.BytesIO(data)), dtype=np.float32)

    # ------------------------------------------------------------------
    # Convenience: NAC stamp + LOLA elevation for a bbox
    # ------------------------------------------------------------------

    def fetch_nac_chip(
        self,
        bbox: tuple[float, float, float, float],
        width_px: int = 512,
        height_px: int = 512,
    ) -> np.ndarray | None:
        """Fetch an NAC grayscale chip as a float32 array in [0, 1]."""
        img = self.get_map_png(NAC_STAMP_LAYER, bbox, width_px, height_px)
        if img is None:
            return None
        gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        return gray

    def fetch_lola_elevation(
        self,
        bbox: tuple[float, float, float, float],
        width_px: int = 512,
        height_px: int = 512,
    ) -> np.ndarray | None:
        """Fetch LOLA DTM elevation [m] for a bbox."""
        return self.get_map_float_tiff(LOLA_DTM_LAYER, bbox, width_px, height_px)
