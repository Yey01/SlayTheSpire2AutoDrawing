import numpy as np


def AABBDistances(points, vertices, line_segments):
    """Pure NumPy fallback for edge_distance_aabb.AABBDistances.

    The upstream package is a compiled extension. This implementation is slower,
    but keeps inference runnable on Python versions without a prebuilt wheel.
    """
    points = np.asarray(points, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    line_segments = np.asarray(line_segments, dtype=np.int64)

    seg_a = vertices[line_segments[:, 0]]
    seg_b = vertices[line_segments[:, 1]]
    seg_v = seg_b - seg_a
    seg_len2 = np.sum(seg_v * seg_v, axis=1)
    seg_len2[seg_len2 <= 1e-12] = 1e-12

    best_dist2 = np.full(points.shape[0], np.inf, dtype=np.float64)
    best_pt = np.zeros((points.shape[0], 2), dtype=np.float64)

    point_chunk = 4096
    segment_chunk = 512
    for ps in range(0, points.shape[0], point_chunk):
        pe = min(ps + point_chunk, points.shape[0])
        p = points[ps:pe]
        local_best_dist2 = np.full(p.shape[0], np.inf, dtype=np.float64)
        local_best_pt = np.zeros((p.shape[0], 2), dtype=np.float64)

        for ss in range(0, seg_a.shape[0], segment_chunk):
            se = min(ss + segment_chunk, seg_a.shape[0])
            a = seg_a[ss:se]
            v = seg_v[ss:se]
            denom = seg_len2[ss:se]

            pa = p[:, None, :] - a[None, :, :]
            t = np.sum(pa * v[None, :, :], axis=2) / denom[None, :]
            t = np.clip(t, 0.0, 1.0)
            closest = a[None, :, :] + t[:, :, None] * v[None, :, :]
            diff = p[:, None, :] - closest
            dist2 = np.sum(diff * diff, axis=2)

            idx = np.argmin(dist2, axis=1)
            vals = dist2[np.arange(p.shape[0]), idx]
            update = vals < local_best_dist2
            if np.any(update):
                local_best_dist2[update] = vals[update]
                local_best_pt[update] = closest[np.arange(p.shape[0]), idx][update]

        best_dist2[ps:pe] = local_best_dist2
        best_pt[ps:pe] = local_best_pt

    return np.sqrt(best_dist2).astype(np.float64), best_pt.astype(np.float64)
