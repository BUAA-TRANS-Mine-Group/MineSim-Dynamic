from typing import Dict, List, Tuple, Union, cast

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import shapely.geometry as geom

from devkit.common.actor_state.ego_state import EgoState
from devkit.common.actor_state.state_representation import Point2D, StateSE2



def extract_polygon_from_map_object(map_object: MapObject) -> List[Point2D]:
    """
    Extract polygon from map object.
    :param map_object: input MapObject.
    :return: polygon as list of Point2D.
    """
    x_coords, y_coords = map_object.polygon.exterior.coords.xy
    return [Point2D(x, y) for x, y in zip(x_coords, y_coords)]


def is_in_type(x: float, y: float, vector_layer: VectorLayer) -> bool:
    """
    Checks if position [x, y] is in any entry of type.
    :param x: [m] floating point x-coordinate in global frame.
    :param y: [m] floating point y-coordinate in global frame.
    :param vector_layer: vector layer to be searched through.
    :return True iff position [x, y] is in any entry of type, False if it is not.
    """
    assert vector_layer is not None, "type can not be None!"

    in_polygon = vector_layer.contains(geom.Point(x, y))

    return any(in_polygon.values)


def compute_linestring_heading(linestring: geom.linestring.LineString) -> List[float]:
    """
    Compute the heading of each coordinate to its successor coordinate. The last coordinate will have the same heading
        as the second last coordinate.
    :param linestring: linestring as a shapely LineString.
    :return: a list of headings associated to each starting coordinate.
    """
    coords: npt.NDArray[np.float64] = np.asarray(linestring.coords)
    vectors = np.diff(coords, axis=0)
    angles = np.arctan2(vectors.T[1], vectors.T[0])
    angles = np.append(angles, angles[-1])  # pad end with duplicate heading

    assert len(angles) == len(coords), "Calculated heading must have the same length as input coordinates"

    return list(angles)


def compute_curvature(point1: geom.Point, point2: geom.Point, point3: geom.Point) -> float:
    """
    Estimate signed curvature along the three points.
    :param point1: First point of a circle.
    :param point2: Second point of a circle.
    :param point3: Third point of a circle.
    :return signed curvature of the three points.
    """
    # points_utm is a 3-by-2 array, containing the easting and northing coordinates of 3 points
    # Compute distance to each point
    a = point1.distance(point2)
    b = point2.distance(point3)
    c = point3.distance(point1)

    # Compute inverse radius of circle using surface of triangle (for which Heron's formula is used)
    surface_2 = (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))

    if surface_2 < 1e-6:
        # In this case the points are almost aligned in a lane
        return 0.0

    assert surface_2 >= 0
    k = np.sqrt(surface_2) / 4  # Heron's formula for triangle's surface
    den = a * b * c  # Denumerator; make sure there is no division by zero.
    curvature = 4 * k / den if not np.isclose(den, 0.0) else 0.0

    # The curvature is unsigned, in order to extract sign, the third point is checked wrt to point1-point2 line
    position = np.sign((point2.x - point1.x) * (point3.y - point1.y) - (point2.y - point1.y) * (point3.x - point1.x))

    return float(position * curvature)


def get_distance_between_map_object_and_point(point: Point2D, map_object: MapObject) -> float:
    """
    Get distance between point and nearest surface of specified map object.
    :param point: Point to calculate distance between.
    :param map_object: MapObject (containing underlying polygon) to check distance between.
    :return: Computed distance.
    """
    return float(geom.Point(point.x, point.y).distance(map_object.polygon))


def extract_discrete_polyline(polyline: geom.LineString) -> List[StateSE2]:
    """
    Returns a discretized polyline composed of StateSE2 as nodes.
    :param polyline: the polyline of interest.
    :returns: linestring as a list of waypoints represented by StateSE2.
    """
    assert polyline.length > 0.0, "The length of the polyline has to be greater than 0!"

    headings = compute_linestring_heading(polyline)
    x_coords, y_coords = polyline.coords.xy

    return [StateSE2(x, y, heading) for x, y, heading in zip(x_coords, y_coords, headings)]


def estimate_curvature_along_path(path: geom.LineString, arc_length: float, distance_for_curvature_estimation: float) -> float:
    """
    Estimate curvature along a path at arc_length from origin.
    :param path: LineString creating a continuous path.
    :param arc_length: [m] distance from origin of the path.
    :param distance_for_curvature_estimation: [m] the distance used to construct 3 points.
    :return estimated curvature at point arc_length.
    """
    assert 0 <= arc_length <= path.length

    # Extract 3 points from a path
    if path.length < 2.0 * distance_for_curvature_estimation:
        # In this case the arch_length is too short
        first_arch_length = 0.0
        second_arc_length = path.length / 2.0
        third_arc_length = path.length
    elif arc_length - distance_for_curvature_estimation < 0.0:
        # In this case the arch_length is too close to origin
        first_arch_length = 0.0
        second_arc_length = distance_for_curvature_estimation
        third_arc_length = 2.0 * distance_for_curvature_estimation
    elif arc_length + distance_for_curvature_estimation > path.length:
        # In this case the arch_length is too close to end of the path
        first_arch_length = path.length - 2.0 * distance_for_curvature_estimation
        second_arc_length = path.length - distance_for_curvature_estimation
        third_arc_length = path.length
    else:  # In this case the arc_length lands along the path
        first_arch_length = arc_length - distance_for_curvature_estimation
        second_arc_length = arc_length
        third_arc_length = arc_length + distance_for_curvature_estimation

    first_arch_position = path.interpolate(first_arch_length)
    second_arch_position = path.interpolate(second_arc_length)
    third_arch_position = path.interpolate(third_arc_length)

    return compute_curvature(first_arch_position, second_arch_position, third_arch_position)
