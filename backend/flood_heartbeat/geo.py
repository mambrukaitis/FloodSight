from __future__ import annotations

from dataclasses import dataclass

from pyproj import Transformer
from shapely import transform
from shapely.geometry import Point

GEOGRAPHIC_CRS = "EPSG:4326"
METRIC_CRS = "EPSG:32617"


@dataclass(frozen=True)
class StudyArea:
    latitude: float = 29.6516
    longitude: float = -82.3248
    radius_km: float = 30.0

    def __post_init__(self) -> None:
        if not -90 <= self.latitude <= 90:
            raise ValueError("Latitude must be between -90 and 90.")
        if not -180 <= self.longitude <= 180:
            raise ValueError("Longitude must be between -180 and 180.")
        if self.radius_km <= 0:
            raise ValueError("Radius must be greater than zero.")

    @property
    def to_metric(self) -> Transformer:
        return Transformer.from_crs(GEOGRAPHIC_CRS, METRIC_CRS, always_xy=True)

    @property
    def to_geographic(self) -> Transformer:
        return Transformer.from_crs(METRIC_CRS, GEOGRAPHIC_CRS, always_xy=True)

    @property
    def center_metric(self) -> Point:
        x, y = self.to_metric.transform(self.longitude, self.latitude)
        return Point(x, y)

    @property
    def circle_metric(self):
        return self.center_metric.buffer(self.radius_km * 1000, quad_segs=64)

    @property
    def bbox_geographic(self) -> tuple[float, float, float, float]:
        return transform(
            self.circle_metric,
            self.to_geographic.transform,
            interleaved=False,
        ).bounds

    def distance_km(self, latitude: float, longitude: float) -> float:
        x, y = self.to_metric.transform(longitude, latitude)
        return self.center_metric.distance(Point(x, y)) / 1000

    def project(self, geometry):
        return transform(
            geometry,
            self.to_metric.transform,
            interleaved=False,
        )

    def unproject(self, geometry):
        return transform(
            geometry,
            self.to_geographic.transform,
            interleaved=False,
        )


def is_geographic(geometry) -> bool:
    min_x, min_y, max_x, max_y = geometry.bounds
    return (
        -180 <= min_x <= 180
        and -180 <= max_x <= 180
        and -90 <= min_y <= 90
        and -90 <= max_y <= 90
    )
