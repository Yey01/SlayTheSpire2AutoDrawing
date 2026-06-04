# -*- coding: utf-8 -*-

import json
import math
import os
import time

import cv2
import numpy as np
from skimage.morphology import skeletonize


class PathPipelineMixin:
    def _build_endpoint_grid(self, paths, cell_size):
        grid = {}
        for path_idx, path in enumerate(paths):
            if len(path) < 2:
                continue
            for at_start, point in [(True, path[0]), (False, path[-1])]:
                cx = int(point[0] // cell_size)
                cy = int(point[1] // cell_size)
                grid.setdefault((cx, cy), []).append((path_idx, at_start, point))
        return grid

    def _query_endpoint_grid(self, grid, point, cell_size, radius_cells=1):
        cx = int(point[0] // cell_size)
        cy = int(point[1] // cell_size)
        candidates = []
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                candidates.extend(grid.get((cx + dx, cy + dy), []))
        return candidates

    def _find_best_onnx_grid_merge(self, path_idx, path, paths, grid, cell_size,
                                     density_map, gap, angle, used):
        if len(path) < 2:
            return None
        best_score = float("inf")
        best_result = None

        for at_start, pt in [(True, path[0]), (False, path[-1])]:
            tang_i = self._endpoint_tangent(path, at_start)
            candidates = self._query_endpoint_grid(grid, pt, cell_size)

            for j, at_start_j, pt_j in candidates:
                if j == path_idx or used[j]:
                    continue
                other = paths[j]
                if len(other) < 2:
                    continue
                dist = math.hypot(pt[0] - pt_j[0], pt[1] - pt_j[1])
                if dist > gap:
                    continue

                if self.onnx_merge_skip_density and density_map is not None:
                    if self._is_high_density_endpoint(pt, density_map):
                        if dist > max(3, gap // 2):
                            continue

                tang_j = self._endpoint_tangent(other, at_start_j)
                angle_i = self._angle_between(tang_i, (-tang_i[0], -tang_i[1]))
                angle_j = self._angle_between(tang_j, (-tang_j[0], -tang_j[1]))

                if at_start == at_start_j:
                    angle_merge = self._angle_between(tang_i, (-tang_j[0], -tang_j[1]))
                else:
                    angle_merge = self._angle_between(tang_i, tang_j)

                if angle_merge > angle:
                    continue

                if len(candidates) > self.onnx_merge_max_candidates * 2:
                    candidate_count = 0
                    for c in candidates:
                        if c[0] != path_idx and not used[c[0]] and len(paths[c[0]]) >= 2:
                            candidate_count += 1
                    if candidate_count > self.onnx_merge_max_candidates:
                        continue

                score = dist + angle_i * 0.05 + angle_j * 0.05
                if score < best_score:
                    best_score = score
                    best_result = (j, at_start, at_start_j, pt_j)

        if best_result is not None:
            j, at_start, at_start_j, pt_j = best_result
            other = paths[j]
            result = self._try_merge_two_paths(path, other, gap, angle)
            if result is not None:
                return (j, result)
        return None

    def merge_onnx_paths_grid(self, paths, image_shape, density_map=None):
        if not self.onnx_merge_enabled or len(paths) < 2:
            return paths

        gap = self.onnx_merge_gap
        angle = self.onnx_merge_angle
        cell_size = max(gap * 2, 8)
        merged = list(paths)
        stats = {
            "enabled": True,
            "input_paths": len(paths),
            "gap": gap,
            "angle": angle,
            "rounds": [],
            "merged_pairs": 0,
        }

        for round_idx in range(self.onnx_merge_rounds):
            grid = self._build_endpoint_grid(merged, cell_size)
            used = [False] * len(merged)
            next_paths = []
            round_merges = 0

            for i, path in enumerate(merged):
                if used[i]:
                    continue
                result = self._find_best_onnx_grid_merge(
                    i, path, merged, grid, cell_size, density_map, gap, angle, used
                )
                if result is not None:
                    j, merged_path = result
                    used[i] = True
                    used[j] = True
                    next_paths.append(merged_path)
                    round_merges += 1
                else:
                    used[i] = True
                    next_paths.append(path)

            merged = next_paths
            stats["rounds"].append({
                "round": round_idx,
                "input": len(next_paths) if round_idx == 0 else stats["rounds"][-1]["output"],
                "output": len(next_paths),
                "merges": round_merges,
            })
            stats["merged_pairs"] += round_merges
            if round_merges == 0:
                break

        self.debug_onnx_merge_stats = stats
        return merged

    def should_enable_onnx_cover(self, clean_strategy, line_raw):
        mode = getattr(self, "onnx_cover_mode", "auto")
        dark_ratio = 0.0
        if line_raw is not None:
            dark_ratio = np.count_nonzero(line_raw < 220) / line_raw.size
        if not self.onnx_cover_enabled or mode == "off":
            enabled = False
        elif mode == "always":
            enabled = True
        else:
            enabled = (
                clean_strategy in self.onnx_cover_enable_strategies
                or dark_ratio > self.onnx_cover_dark_ratio_threshold
            )
        self.debug_onnx_cover_decision = {
            "enabled": enabled,
            "mode": mode,
            "clean_strategy": clean_strategy,
            "dark_ratio_lt220": round(float(dark_ratio), 4) if line_raw is not None else None,
            "strategy_allowed": clean_strategy in self.onnx_cover_enable_strategies if clean_strategy else False,
            "dark_ratio_threshold": self.onnx_cover_dark_ratio_threshold,
        }
        return enabled

    def split_thin_thick_onnx_mask(self, clean_mask):
        dist = cv2.distanceTransform(clean_mask, cv2.DIST_L2, 3)
        thick_mask = (dist >= self.onnx_cover_width_threshold).astype(np.uint8) * 255
        thin_mask = cv2.bitwise_and(clean_mask, cv2.bitwise_not(thick_mask))
        return thin_mask, thick_mask, dist

    def _estimate_pca_direction(self, region_ys, region_xs, left, top):
        if len(region_ys) < 4:
            return None
        pts = np.column_stack([region_xs.astype(np.float32), region_ys.astype(np.float32)])
        mean = pts.mean(axis=0)
        cov = np.cov((pts - mean).T)
        if cov.shape != (2, 2) or np.isnan(cov).any():
            return None
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            return None
        main_vec = eigvecs[:, np.argmax(eigvals)]
        normal_vec = np.array([-main_vec[1], main_vec[0]], dtype=np.float32)
        angle_deg = float(math.degrees(math.atan2(main_vec[1], main_vec[0])))
        return main_vec, normal_vec, angle_deg

    def _generate_pca_cover_for_region(self, region_ys, region_xs, left, top,
                                         area, confidence, spacing):
        result = self._estimate_pca_direction(region_ys, region_xs, left, top)
        if result is None:
            return self._generate_bbox_cover_for_region(
                region_ys, region_xs, left, top, area, confidence, spacing)
        main_vec, normal_vec, angle_deg = result

        pts = np.column_stack([region_xs.astype(np.float32), region_ys.astype(np.float32)])
        mean = pts.mean(axis=0)

        proj_normal = (pts - mean).dot(normal_vec)
        proj_main = (pts - mean).dot(main_vec)
        n_min, n_max = proj_normal.min(), proj_normal.max()
        m_min, m_max = proj_main.min(), proj_main.max()

        paths = []
        n = n_min
        while n <= n_max:
            band = (proj_normal >= n) & (proj_normal < n + spacing)
            if not np.any(band):
                n += spacing
                continue
            band_pts = proj_main[band]
            if len(band_pts) < 2:
                n += spacing
                continue
            m0 = band_pts.min()
            m1 = band_pts.max()
            seg_len = abs(m1 - m0)
            if seg_len < self.onnx_cover_min_segment_length:
                n += spacing
                continue
            pt_center = mean + n * normal_vec
            p0 = pt_center + m0 * main_vec
            p1 = pt_center + m1 * main_vec
            paths.append([[int(round(p0[0])), int(round(p0[1]))],
                          [int(round(p1[0])), int(round(p1[1]))]])
            n += spacing
        return paths

    def _generate_bbox_cover_for_region(self, region_ys, region_xs, left, top,
                                          area, confidence, spacing):
        if len(region_ys) < 4:
            return []
        # region_xs/region_ys are already in global image coordinates
        x_vals = region_xs
        y_vals = region_ys
        w_range = x_vals.max() - x_vals.min() + 1
        h_range = y_vals.max() - y_vals.min() + 1
        paths = []

        if w_range > h_range * 1.5:
            for cx in range(x_vals.min(), x_vals.max() + 1, spacing):
                col_mask = (x_vals >= cx) & (x_vals < cx + spacing)
                if not np.any(col_mask):
                    continue
                cy = y_vals[col_mask]
                if len(cy) < 2:
                    continue
                paths.append([(cx + spacing // 2, int(cy.min())),
                              (cx + spacing // 2, int(cy.max()))])
        elif h_range > w_range * 1.5:
            for ry in range(y_vals.min(), y_vals.max() + 1, spacing):
                row_mask = (y_vals >= ry) & (y_vals < ry + spacing)
                if not np.any(row_mask):
                    continue
                rx = x_vals[row_mask]
                if len(rx) < 2:
                    continue
                paths.append([(int(rx.min()), ry + spacing // 2),
                              (int(rx.max()), ry + spacing // 2)])
        else:
            diag_spacing = spacing * 2
            for offset in range(-max(w_range, h_range), max(w_range, h_range), diag_spacing):
                dpoints = []
                for t in range(max(w_range, h_range)):
                    px = x_vals.min() + offset + t
                    py = y_vals.min() + t
                    if x_vals.min() <= px <= x_vals.max() and y_vals.min() <= py <= y_vals.max():
                        idx = np.where((x_vals == px) & (y_vals == py))[0]
                        if len(idx) > 0:
                            dpoints.append((px, py))
                if len(dpoints) >= 2:
                    paths.append([dpoints[0], dpoints[-1]])
        return paths

    def generate_onnx_stroke_cover_paths(self, thick_mask, line_raw=None):
        if not self.onnx_cover_enabled:
            return [], []

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thick_mask, connectivity=8)
        spacing = max(self.onnx_cover_spacing, 2)
        min_conf = self.onnx_cover_min_raw_confidence
        min_score = self.onnx_cover_min_region_score
        use_pca = (self.onnx_cover_direction_mode == "pca")

        rejected_area = 0
        rejected_conf = 0
        rejected_score = 0
        rejected_size = 0

        regions = []
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.onnx_cover_min_area:
                rejected_area += 1
                continue

            left = stats[i, cv2.CC_STAT_LEFT]
            top = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]

            region_mask = (labels[top:top + bh, left:left + bw] == i)
            region_ys, region_xs = np.where(region_mask)
            if len(region_ys) < 4:
                rejected_size += 1
                continue
            region_ys = region_ys + top
            region_xs = region_xs + left

            confidence = 1.0
            if line_raw is not None:
                raw_vals = line_raw[region_ys, region_xs].astype(np.float32)
                p35_val = float(np.percentile(raw_vals, 35))
                confidence = float(1.0 - p35_val / 255.0)
            if confidence < min_conf:
                rejected_conf += 1
                continue

            score = area * confidence
            if score < min_score:
                rejected_score += 1
                continue

            regions.append({
                "label_id": i, "area": area, "confidence": round(confidence, 4),
                "score": round(score, 2), "left": int(left), "top": int(top),
                "ys": region_ys, "xs": region_xs,
            })

        fallback_applied = False
        if len(regions) == 0 and (num_labels - 1) > 0:
            decision = getattr(self, "debug_onnx_cover_decision", None)
            if decision and decision.get("enabled"):
                fallback_min_conf = 0.20
                fallback_applied = True
                for i in range(1, num_labels):
                    area = int(stats[i, cv2.CC_STAT_AREA])
                    if area < self.onnx_cover_min_area:
                        continue
                    left = stats[i, cv2.CC_STAT_LEFT]
                    top = stats[i, cv2.CC_STAT_TOP]
                    bw = stats[i, cv2.CC_STAT_WIDTH]
                    bh = stats[i, cv2.CC_STAT_HEIGHT]
                    region_mask = (labels[top:top + bh, left:left + bw] == i)
                    region_ys, region_xs = np.where(region_mask)
                    if len(region_ys) < 4:
                        continue
                    region_ys = region_ys + top
                    region_xs = region_xs + left
                    confidence = 1.0
                    if line_raw is not None:
                        raw_vals = line_raw[region_ys, region_xs].astype(np.float32)
                        p35_val = float(np.percentile(raw_vals, 35))
                        confidence = float(1.0 - p35_val / 255.0)
                    if confidence < fallback_min_conf:
                        continue
                    score = area * confidence
                    regions.append({
                        "label_id": i, "area": area, "confidence": round(confidence, 4),
                        "score": round(score, 2), "left": int(left), "top": int(top),
                        "ys": region_ys, "xs": region_xs,
                    })

        regions.sort(key=lambda r: r["score"], reverse=True)

        cover_paths = []
        cover_metas = []
        total_paths = 0
        accepted_regions = 0
        hit_global_limit = False
        top_region_summary = []

        for region in regions:
            if total_paths >= self.onnx_cover_max_paths:
                hit_global_limit = True
                break

            if use_pca:
                region_paths = self._generate_pca_cover_for_region(
                    region["ys"], region["xs"],
                    region["left"], region["top"],
                    region["area"], region["confidence"], spacing)
            else:
                region_paths = self._generate_bbox_cover_for_region(
                    region["ys"], region["xs"],
                    region["left"], region["top"],
                    region["area"], region["confidence"], spacing)

            region_paths = region_paths[:self.onnx_cover_max_paths_per_region]
            if not region_paths:
                continue

            accepted_regions += 1
            if len(top_region_summary) < 10:
                top_region_summary.append({
                    "area": region["area"],
                    "confidence": region["confidence"],
                    "score": region["score"],
                    "paths": len(region_paths),
                })

            for path in region_paths:
                if total_paths >= self.onnx_cover_max_paths:
                    hit_global_limit = True
                    break
                if len(path) < 2:
                    continue
                dx = path[1][0] - path[0][0]
                dy = path[1][1] - path[0][1]
                seg_len = math.hypot(dx, dy)
                if seg_len < self.onnx_cover_min_segment_length:
                    continue
                cover_paths.append([(path[0][0], path[0][1]), (path[1][0], path[1][1])])
                cover_metas.append({
                    "kind": "cover", "level": "detail",
                    "source": "onnx_cover",
                    "region_area": region["area"],
                    "length": float(seg_len),
                    "curvature": 0.0,
                })
                total_paths += 1

        candidate_components = max(0, num_labels - 1)
        self.debug_onnx_cover_stats = {
            "enabled": True,
            "decision_mode": self.onnx_cover_mode,
            "direction_mode": self.onnx_cover_direction_mode,
            "width_threshold": self.onnx_cover_width_threshold,
            "thick_pixels": int(np.count_nonzero(thick_mask)),
            "thick_components": candidate_components,
            "candidate_regions": len(regions) + rejected_area + rejected_conf + rejected_score + rejected_size,
            "accepted_regions": accepted_regions,
            "rejected_by_area": rejected_area,
            "rejected_by_confidence": rejected_conf,
            "rejected_by_score": rejected_score,
            "rejected_by_size": rejected_size,
            "cover_paths": len(cover_paths),
            "max_paths": self.onnx_cover_max_paths,
            "max_paths_per_region": self.onnx_cover_max_paths_per_region,
            "min_raw_confidence": min_conf,
            "min_region_score": min_score,
            "spacing": spacing,
            "hit_global_limit": hit_global_limit,
            "fallback_applied": fallback_applied,
        }
        if self.debug_output:
            import json as _json
            regions_debug = {
                "candidate_regions": self.debug_onnx_cover_stats["candidate_regions"],
                "accepted_regions": accepted_regions,
                "rejected_by_area": rejected_area,
                "rejected_by_confidence": rejected_conf,
                "rejected_by_score": rejected_score,
                "rejected_by_size": rejected_size,
                "top_regions": top_region_summary,
            }
            debug_dir = self.get_debug_output_dir("fast", onnx=True)
            with open(os.path.join(debug_dir, "debug_onnx_cover_regions.json"), "w") as f:
                _json.dump(regions_debug, f, indent=2, ensure_ascii=False)
        return cover_paths, cover_metas

    def build_ink_mask(self, gray):
        thresh = self.threshold_var.get()
        block_size = max(3, thresh // 10 * 2 + 1)
        if block_size % 2 == 0:
            block_size += 1
        c_val = max(2, thresh // 10)

        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, c_val
        )
        _, simple = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
        ink_mask = cv2.bitwise_or(adaptive, simple)
        return ink_mask

    def preprocess_line_mask(self, ink_mask):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
        clean_mask = np.zeros_like(ink_mask)
        min_area = max(4, self.min_len_var.get() // 3)

        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                clean_mask[labels == i] = 255

        close_kernel = np.ones((2, 2), np.uint8)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, close_kernel)
        return clean_mask

    def skeletonize_mask(self, clean_mask):
        binary = clean_mask > 0
        skeleton = skeletonize(binary)
        return (skeleton * 255).astype(np.uint8)

    def _count_neighbors(self, skeleton, x, y):
        h, w = skeleton.shape
        count = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and skeleton[ny, nx] > 0:
                    count += 1
        return count

    def _get_unvisited_neighbors(self, skeleton, x, y, visited, junction_set):
        h, w = skeleton.shape
        neighbors = []
        junction_neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and skeleton[ny, nx] > 0:
                    if (nx, ny) in junction_set:
                        if not visited[ny, nx]:
                            junction_neighbors.append((nx, ny))
                    elif not visited[ny, nx]:
                        neighbors.append((nx, ny))
        if neighbors:
            return neighbors
        return junction_neighbors

    def _trace_from(self, skeleton, start_x, start_y, visited, junction_set):
        path = [(start_x, start_y)]
        visited[start_y, start_x] = True
        current_x, current_y = start_x, start_y

        while True:
            next_pixels = self._get_unvisited_neighbors(
                skeleton, current_x, current_y, visited, junction_set
            )
            if not next_pixels:
                break
            nx, ny = next_pixels[0]
            path.append((nx, ny))
            visited[ny, nx] = True
            current_x, current_y = nx, ny
            if (current_x, current_y) in junction_set:
                break
        return path

    def _trace_loop(self, skeleton, start_x, start_y, visited, junction_set):
        path = [(start_x, start_y)]
        visited[start_y, start_x] = True
        current_x, current_y = start_x, start_y

        first_neighbors = self._get_unvisited_neighbors(
            skeleton, current_x, current_y, visited, junction_set
        )
        if not first_neighbors:
            return path

        nx, ny = first_neighbors[0]
        path.append((nx, ny))
        visited[ny, nx] = True
        current_x, current_y = nx, ny

        while True:
            next_pixels = self._get_unvisited_neighbors(
                skeleton, current_x, current_y, visited, junction_set
            )
            if not next_pixels:
                break
            nx, ny = next_pixels[0]
            if len(path) > 3:
                if abs(nx - start_x) <= 1 and abs(ny - start_y) <= 1:
                    break
            path.append((nx, ny))
            visited[ny, nx] = True
            current_x, current_y = nx, ny
            if (current_x, current_y) in junction_set:
                break
        return path

    def trace_skeleton_paths(self, skeleton):
        h, w = skeleton.shape
        visited = np.zeros((h, w), dtype=bool)

        ys, xs = np.where(skeleton > 0)
        if len(xs) == 0:
            return []

        endpoints = []
        junctions = []
        for x, y in zip(xs, ys):
            n = self._count_neighbors(skeleton, x, y)
            if n == 1:
                endpoints.append((x, y))
            elif n >= 3:
                junctions.append((x, y))

        junction_set = set(junctions)
        paths = []

        for start_x, start_y in endpoints:
            if visited[start_y, start_x]:
                continue
            path = self._trace_from(skeleton, start_x, start_y, visited, junction_set)
            if path and len(path) >= 2:
                paths.append(path)

        for x, y in zip(xs, ys):
            if not visited[y, x]:
                path = self._trace_loop(skeleton, x, y, visited, junction_set)
                if path and len(path) >= 2:
                    paths.append(path)

        return paths

    def _endpoint_tangent(self, path, at_start):
        window = min(5, len(path) - 1)
        if at_start:
            p0 = path[window]
            p1 = path[0]
        else:
            p0 = path[-window - 1]
            p1 = path[-1]
        return (p1[0] - p0[0], p1[1] - p0[1])

    def _angle_between(self, v1, v2):
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        if mag1 < 1e-6 or mag2 < 1e-6:
            return 180.0
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        return math.degrees(math.acos(cos_angle))

    def _try_merge_two_paths(self, path_a, path_b, merge_gap, merge_angle):
        if len(path_a) < 2 or len(path_b) < 2:
            return None

        tan_a_end = self._endpoint_tangent(path_a, at_start=False)
        tan_a_start = self._endpoint_tangent(path_a, at_start=True)
        tan_b_end = self._endpoint_tangent(path_b, at_start=False)
        tan_b_start = self._endpoint_tangent(path_b, at_start=True)

        candidates = []
        # A.end -> B.start
        dist = math.hypot(path_a[-1][0] - path_b[0][0], path_a[-1][1] - path_b[0][1])
        if dist <= merge_gap:
            # direction A.end should be roughly opposite to B.start for a smooth join
            # Actually we want A's exit direction to match B's entry direction
            angle = self._angle_between(tan_a_end, tan_b_start)
            if angle <= merge_angle:
                candidates.append((dist, angle, False, False))

        # A.end -> B.end
        dist = math.hypot(path_a[-1][0] - path_b[-1][0], path_a[-1][1] - path_b[-1][1])
        if dist <= merge_gap:
            angle = self._angle_between(tan_a_end, (-tan_b_end[0], -tan_b_end[1]))
            if angle <= merge_angle:
                candidates.append((dist, angle, False, True))

        # A.start -> B.start
        dist = math.hypot(path_a[0][0] - path_b[0][0], path_a[0][1] - path_b[0][1])
        if dist <= merge_gap:
            angle = self._angle_between((-tan_a_start[0], -tan_a_start[1]), tan_b_start)
            if angle <= merge_angle:
                candidates.append((dist, angle, True, False))

        # A.start -> B.end
        dist = math.hypot(path_a[0][0] - path_b[-1][0], path_a[0][1] - path_b[-1][1])
        if dist <= merge_gap:
            angle = self._angle_between((-tan_a_start[0], -tan_a_start[1]), (-tan_b_end[0], -tan_b_end[1]))
            if angle <= merge_angle:
                candidates.append((dist, angle, True, True))

        if not candidates:
            return None

        candidates.sort(key=lambda x: (x[0], x[1]))
        _, _, reverse_a, reverse_b = candidates[0]

        if reverse_a:
            path_a = path_a[::-1]
        if reverse_b:
            path_b = path_b[::-1]

        # Check that merged path won't have sharp U-turn by checking angle at connection
        connection_vec = (path_b[0][0] - path_a[-1][0], path_b[0][1] - path_a[-1][1])
        conn_angle = self._angle_between(tan_a_end if not reverse_a else (-tan_a_start[0], -tan_a_start[1]), connection_vec)
        if conn_angle > merge_angle * 1.5:
            return None

        return path_a + path_b

    def _is_high_density_endpoint(self, point, density_map):
        if density_map is None:
            return False
        x, y = point
        h, w = density_map.shape
        r = 8
        y1 = max(0, y - r)
        y2 = min(h, y + r)
        x1 = max(0, x - r)
        x2 = min(w, x + r)
        if y2 > y1 and x2 > x1:
            patch = density_map[y1:y2, x1:x2]
            return float(np.mean(patch)) > 0.15 if patch.size > 0 else False
        return False

    def merge_nearby_paths(self, paths, density_map=None):
        if len(paths) < 2:
            return paths

        img_w, img_h = self.image_size
        merge_gap_normal = max(6, min(12, int(min(img_w, img_h) * 0.020)))
        merge_angle_normal = 42.0
        merge_rounds_normal = 5

        merge_gap_dense = max(3, min(6, int(min(img_w, img_h) * 0.010)))
        merge_angle_dense = 25.0
        merge_rounds_dense = 3

        merged = list(paths)
        for round_idx in range(merge_rounds_normal):
            if len(merged) < 2:
                break
            any_merged = False
            n = len(merged)
            used = [False] * n
            new_merged = []

            for i in range(n):
                if used[i]:
                    continue

                endpoint_i_dense = density_map is not None and (
                    self._is_high_density_endpoint(merged[i][0], density_map) or
                    self._is_high_density_endpoint(merged[i][-1], density_map)
                )

                best_j = -1
                best_result = None
                for j in range(n):
                    if i == j or used[j]:
                        continue

                    endpoint_j_dense = density_map is not None and (
                        self._is_high_density_endpoint(merged[j][0], density_map) or
                        self._is_high_density_endpoint(merged[j][-1], density_map)
                    )

                    if endpoint_i_dense or endpoint_j_dense:
                        if round_idx >= merge_rounds_dense:
                            continue
                        gap = merge_gap_dense
                        angle = merge_angle_dense
                    else:
                        gap = merge_gap_normal
                        angle = merge_angle_normal

                    result = self._try_merge_two_paths(merged[i], merged[j], gap, angle)
                    if result is not None:
                        best_j = j
                        best_result = result
                        break

                if best_j >= 0:
                    used[i] = True
                    used[best_j] = True
                    new_merged.append(best_result)
                    any_merged = True
                else:
                    used[i] = True
                    new_merged.append(merged[i])

            merged = new_merged
            if not any_merged:
                break

        return merged

    def bridge_small_gaps(self, paths):
        self.debug_bridge_segments = []
        if len(paths) < 2:
            return paths

        img_w, img_h = self.image_size
        bridge_gap = max(3, min(6, int(min(img_w, img_h) * 0.01)))
        bridge_angle = 25.0

        n = len(paths)
        used_in_bridge = [False] * n
        bridged = []

        for i in range(n):
            if used_in_bridge[i]:
                continue
            best_j = -1
            best_dist = float('inf')
            best_combined = None

            for j in range(n):
                if i == j or used_in_bridge[j]:
                    continue

                candidates = []
                for rev_i, rev_j in [(False, False), (False, True), (True, False), (True, True)]:
                    pi = paths[i][::-1] if rev_i else list(paths[i])
                    pj = paths[j][::-1] if rev_j else list(paths[j])

                    dist = math.hypot(pi[-1][0] - pj[0][0], pi[-1][1] - pj[0][1])
                    if dist <= bridge_gap:
                        tan_i = self._endpoint_tangent(paths[i], at_start=rev_i)
                        tan_j = self._endpoint_tangent(paths[j], at_start=not rev_j)
                        if rev_i:
                            tan_i = (-tan_i[0], -tan_i[1])
                        if rev_j:
                            tan_j = (-tan_j[0], -tan_j[1])

                        conn_vec = (pj[0][0] - pi[-1][0], pj[0][1] - pi[-1][1])
                        angle_i = self._angle_between(tan_i, conn_vec)
                        angle_j = self._angle_between(conn_vec, tan_j)

                        if angle_i <= bridge_angle and angle_j <= bridge_angle:
                            candidates.append((dist, rev_i, rev_j))

                if candidates:
                    candidates.sort(key=lambda x: x[0])
                    dist, rev_i, rev_j = candidates[0]
                    if dist < best_dist:
                        best_dist = dist
                        best_j = j
                        pi = paths[i][::-1] if rev_i else list(paths[i])
                        pj = paths[j][::-1] if rev_j else list(paths[j])
                        gap_dist = max(1, int(best_dist))
                        gap_points = []
                        dx = pj[0][0] - pi[-1][0]
                        dy = pj[0][1] - pi[-1][1]
                        for k in range(1, gap_dist):
                            gap_points.append(
                                (int(round(pi[-1][0] + dx * k / gap_dist)),
                                 int(round(pi[-1][1] + dy * k / gap_dist)))
                            )
                        best_combined = pi + gap_points + pj

            if best_j >= 0 and best_combined is not None:
                used_in_bridge[i] = True
                used_in_bridge[best_j] = True
                bridged.append(best_combined)
                if gap_points:
                    self.debug_bridge_segments.append(gap_points)
            elif not used_in_bridge[i]:
                used_in_bridge[i] = True
                bridged.append(list(paths[i]))

        return bridged

    def _estimate_curvature(self, path):
        if len(path) < 3:
            return 0.0
        total_angle = 0.0
        for i in range(1, len(path) - 1):
            x1, y1 = path[i - 1]
            x2, y2 = path[i]
            x3, y3 = path[i + 1]
            v1 = (x2 - x1, y2 - y1)
            v2 = (x3 - x2, y3 - y2)
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
            if mag1 > 0 and mag2 > 0:
                cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
                total_angle += math.acos(cos_angle)
        return total_angle / (len(path) - 2)

    def _path_length(self, path):
        if len(path) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(path)):
            total += math.hypot(path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
        return total

    def simplify_path_adaptive(self, path):
        if len(path) < 3:
            return path
        length = self._path_length(path)
        curvature = self._estimate_curvature(path)

        contour = np.array([[[p[0], p[1]]] for p in path], dtype=np.int32)

        if length > 120 and curvature < 0.3:
            epsilon = length * 0.008
        elif length > 30:
            epsilon = length * 0.003
        else:
            epsilon = length * 0.001

        epsilon = max(0.5, min(epsilon, 5.0))
        simplified = cv2.approxPolyDP(contour, epsilon, False)
        return [(int(p[0][0]), int(p[0][1])) for p in simplified]

    def build_path_density_map(self, paths, image_shape):
        h, w = image_shape
        density = np.zeros((h, w), dtype=np.float32)
        for path in paths:
            for i in range(len(path) - 1):
                x1, y1 = path[i]
                x2, y2 = path[i + 1]
                cv2.line(density, (x1, y1), (x2, y2), 1.0, 1)
        kernel = np.ones((15, 15), np.float32)
        density = cv2.filter2D(density, -1, kernel)
        return density

    def classify_path_v2(self, path, density_map=None):
        length = self._path_length(path)
        curvature = self._estimate_curvature(path)

        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        bbox_w = max(xs) - min(xs) if xs else 0
        bbox_h = max(ys) - min(ys) if ys else 0
        bbox_area = bbox_w * bbox_h

        local_density = 0.0
        if density_map is not None and len(path) >= 2:
            h, w = density_map.shape
            cx = int(sum(xs) / len(xs)) if xs else 0
            cy = int(sum(ys) / len(ys)) if ys else 0
            r = max(bbox_w, bbox_h) // 2 + 5
            y1 = max(0, cy - r)
            y2 = min(h, cy + r)
            x1 = max(0, cx - r)
            x2 = min(w, cx + r)
            if y2 > y1 and x2 > x1:
                patch = density_map[y1:y2, x1:x2]
                local_density = float(np.mean(patch)) if patch.size > 0 else 0.0

        in_dense_region = local_density > 0.15

        if length > 220:
            return "outline"
        if length > 140 and bbox_area > 2500:
            return "outline"
        if length > 100 and bbox_area > 5000 and curvature < 0.55:
            return "outline"

        if in_dense_region and length < 60:
            return "detail"
        if length < 15 or bbox_area < 150:
            return "detail"

        return "normal"

    def build_tone_mask(self, gray):
        dark_thresh = 110

        _, dark = cv2.threshold(gray, dark_thresh, 255, cv2.THRESH_BINARY_INV)
        block_size = 31
        if block_size % 2 == 0:
            block_size += 1
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, 5
        )

        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        local_dark = np.zeros_like(gray)
        local_dark[gray.astype(np.int16) < blur.astype(np.int16) - 12] = 255
        local_dark = local_dark.astype(np.uint8)

        tone_mask = cv2.bitwise_or(dark, adaptive)
        tone_mask = cv2.bitwise_or(tone_mask, local_dark)
        return tone_mask

    def split_line_and_fill_masks(self, gray, ink_mask):
        tone_mask = self.build_tone_mask(gray)
        combined = cv2.bitwise_and(ink_mask, tone_mask)

        blur = cv2.GaussianBlur(gray, (21, 21), 0)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined, connectivity=8)
        fill_mask = np.zeros_like(ink_mask)
        line_mask = ink_mask.copy()

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            bbox_area = bw * bh
            fill_ratio = area / bbox_area if bbox_area > 0 else 0

            component_mask = (labels == i).astype(np.uint8) * 255
            ys, xs = np.where(component_mask > 0)
            if len(ys) > 0:
                mean_gray = float(np.mean(gray[ys, xs]))
                local_bg_ys = np.clip(ys + 10, 0, gray.shape[0] - 1)
                local_bg_xs = np.clip(xs + 10, 0, gray.shape[1] - 1)
                local_bg_gray = float(np.mean(gray[local_bg_ys, local_bg_xs]))
                local_contrast = local_bg_gray - mean_gray
            else:
                mean_gray = 255
                local_contrast = 0

            aspect_ratio = bw / bh if bh > 0 else 1.0

            is_fill = False
            if area >= 10 and fill_ratio >= 0.30 and mean_gray < 145:
                is_fill = True
            elif area >= 5 and fill_ratio >= 0.38 and mean_gray < 125:
                is_fill = True
            elif area >= 8 and fill_ratio >= 0.28 and local_contrast >= 12 and mean_gray < 170:
                is_fill = True

            if is_fill and (aspect_ratio > 8 or aspect_ratio < 0.125):
                if not (area < 30 and mean_gray < 80):
                    is_fill = False

            if is_fill:
                fill_mask[labels == i] = 255
                kernel = np.ones((2, 2), np.uint8)
                eroded = cv2.erode(component_mask, kernel, iterations=1)
                line_mask[eroded > 0] = 0

        return line_mask, fill_mask

    def detect_fill_regions(self, gray, fill_mask):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fill_mask, connectivity=8)
        regions = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 4:
                continue
            region_mask = (labels == i).astype(np.uint8) * 255
            ys, xs = np.where(region_mask > 0)
            mean_gray = float(np.mean(gray[ys, xs])) if len(ys) > 0 else 255
            tone = 1.0 - mean_gray / 255.0
            regions.append({
                "mask": region_mask,
                "area": int(area),
                "bbox": (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                         int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])),
                "centroid": (int(centroids[i][0]), int(centroids[i][1])),
                "mean_gray": mean_gray,
                "tone": tone,
                "aspect_ratio": stats[i, cv2.CC_STAT_WIDTH] / max(1, stats[i, cv2.CC_STAT_HEIGHT])
            })
        return regions

    def generate_hatch_paths(self, region_mask, angle, spacing):
        h, w = region_mask.shape
        angle_rad = math.radians(angle)
        direction = (math.cos(angle_rad), math.sin(angle_rad))
        normal = (-direction[1], direction[0])

        cx, cy = w // 2, h // 2

        paths = []
        max_offset = int(math.sqrt(w**2 + h**2))
        for offset in range(-max_offset, max_offset, spacing):
            scan_points = []
            for t in range(-max_offset, max_offset, 2):
                px = int(cx + normal[0] * offset + direction[0] * t)
                py = int(cy + normal[1] * offset + direction[1] * t)
                if 0 <= px < w and 0 <= py < h and region_mask[py, px] > 0:
                    scan_points.append((px, py))
                else:
                    if len(scan_points) >= 2:
                        paths.append(scan_points)
                    scan_points = []
            if len(scan_points) >= 2:
                paths.append(scan_points)
        return paths

    def generate_spiral_fill_path(self, region_mask):
        h, w = region_mask.shape
        ys, xs = np.where(region_mask > 0)
        if len(ys) == 0:
            return []

        cx = int(np.mean(xs))
        cy = int(np.mean(ys))
        max_r = int(max(np.max(xs) - cx, cx - np.min(xs), np.max(ys) - cy, cy - np.min(ys))) + 5
        spacing = 2

        paths = []
        current_path = []
        theta = 0.0

        while True:
            r = spacing * (theta / (2 * math.pi))
            if r > max_r:
                break
            px = int(cx + r * math.cos(theta))
            py = int(cy + r * math.sin(theta))
            if 0 <= px < w and 0 <= py < h and region_mask[py, px] > 0:
                current_path.append((px, py))
            else:
                if len(current_path) >= 2:
                    paths.append(current_path)
                current_path = []
            theta += 0.15

        if len(current_path) >= 2:
            paths.append(current_path)
        return paths

    def generate_fill_paths(self, fill_regions, gray):
        fill_paths = []
        fill_metas = []
        max_total = 300

        for region in fill_regions:
            if len(fill_paths) >= max_total:
                break

            mask = region["mask"]
            aspect = region["aspect_ratio"]
            tone = region.get("tone", 0.5)
            spacing = max(2, min(7, int(round(7 - tone * 4))))

            if region["area"] < 30 and 0.7 < aspect < 1.4:
                spiral_paths = self.generate_spiral_fill_path(mask)
                for p in spiral_paths:
                    if len(p) >= 2:
                        fill_paths.append(p)
                        fill_metas.append({
                            "kind": "fill", "level": "fill",
                            "source": "spiral", "region_id": id(region),
                            "tone": tone, "fill_strategy": "spiral",
                            "length": self._path_length(p),
                            "curvature": self._estimate_curvature(p)
                        })
            else:
                angle = 0 if aspect > 1.0 else 90
                hatch_paths = self.generate_hatch_paths(mask, angle, spacing)
                for p in hatch_paths:
                    if len(p) >= 2:
                        fill_paths.append(p)
                        fill_metas.append({
                            "kind": "fill", "level": "fill",
                            "source": "hatch", "region_id": id(region),
                            "tone": tone, "fill_strategy": "hatch",
                            "length": self._path_length(p),
                            "curvature": self._estimate_curvature(p)
                        })

        return fill_paths, fill_metas

    def order_draw_tasks(self, line_paths, line_metas, fill_paths, fill_metas):
        if not line_paths and not fill_paths:
            return [], []

        all_paths = []
        all_metas = []

        level_order = {"outline": 0, "normal": 1, "fill": 2, "detail": 3}
        grouped = {"outline": [], "normal": [], "fill": [], "detail": []}

        for path, meta in zip(line_paths, line_metas):
            level = meta if isinstance(meta, str) else meta.get("level", "normal")
            grouped.setdefault(level, []).append((path, meta if not isinstance(meta, str) else {"level": meta, "kind": "line"}))

        for path, meta in zip(fill_paths, fill_metas):
            grouped.setdefault("fill", []).append((path, meta))

        for level in ["outline", "normal", "fill", "detail"]:
            group = grouped.get(level, [])
            if not group:
                continue

            remaining = list(group)
            current_point = (0, 0)

            while remaining:
                best_idx = 0
                min_dist = float('inf')
                reverse_best = False

                for i, (path, meta) in enumerate(remaining):
                    if len(path) < 2:
                        dist_start = (path[0][0] - current_point[0])**2 + (path[0][1] - current_point[1])**2
                        if dist_start < min_dist:
                            min_dist, best_idx, reverse_best = dist_start, i, False
                    else:
                        start_pt = path[0]
                        end_pt = path[-1]
                        dist_start = (start_pt[0] - current_point[0])**2 + (start_pt[1] - current_point[1])**2
                        dist_end = (end_pt[0] - current_point[0])**2 + (end_pt[1] - current_point[1])**2
                        if dist_start < min_dist:
                            min_dist, best_idx, reverse_best = dist_start, i, False
                        if dist_end < min_dist:
                            min_dist, best_idx, reverse_best = dist_end, i, True

                best_path, best_meta = remaining.pop(best_idx)
                if reverse_best:
                    best_path = best_path[::-1]
                all_paths.append(best_path)
                all_metas.append(best_meta)
                if len(best_path) > 0:
                    current_point = best_path[-1]

        return all_paths, all_metas

