#pragma once

#include <vector>
#include <tuple>
#include <set>
#include <algorithm>
#include <cmath>
#include "rectangular_lsap.hpp"

#if !defined(STANDALONE)
#include "edge-impulse-sdk/classifier/ei_classifier_types.h"
#endif

__attribute__((unused)) static bool compare_tuples(std::tuple<int, int, float> a, std::tuple<int, int, float> b) {
    return std::get<2>(a) < std::get<2>(b);
}

float intersection_over_union(const ei_impulse_result_bounding_box_t bbox1, const ei_impulse_result_bounding_box_t bbox2) {
    uint32_t x_left = std::max(bbox1.x, bbox2.x);
    uint32_t y_top = std::max(bbox1.y, bbox2.y);
    uint32_t x_right = std::min(bbox1.x + bbox1.width, bbox2.x + bbox2.width);
    uint32_t y_bottom = std::min(bbox1.y + bbox1.height, bbox2.y + bbox2.height);

    if (x_right < x_left || y_bottom < y_top) {
        return 0.0;
    }

    uint32_t intersection_area = (x_right - x_left) * (y_bottom - y_top);
    uint32_t bbox1_area = bbox1.width * bbox1.height;
    uint32_t bbox2_area = bbox2.width * bbox2.height;

    return static_cast<float>(intersection_area) / static_cast<float>(bbox1_area + bbox2_area - intersection_area);
}

class JonkerVolgenantAlignment {
public:
    JonkerVolgenantAlignment(float iou_threshold) : iou_threshold(iou_threshold) {
    }

    std::vector<std::tuple<int, int, float>> align(const std::vector<ei_impulse_result_bounding_box_t> traces,
                                                   const std::vector<ei_impulse_result_bounding_box_t> detections) {

        if (traces.empty() || detections.empty()) {
            return {};
        }

        std::vector<double> cost_mtx(traces.size() * detections.size());
        for (size_t trace_idx = 0; trace_idx < traces.size(); ++trace_idx) {
            for (size_t detection_idx = 0; detection_idx < detections.size(); ++detection_idx) {
                float iou = intersection_over_union(traces[trace_idx], detections[detection_idx]);
                float cost = 1 - iou;
                EI_LOGD("t_idx=%zu d_idx=%zu cost=%.6f\n", trace_idx, detection_idx, cost);
                cost_mtx[trace_idx * detections.size() + detection_idx] = cost;
            }
        }

        int64_t *alignments_a = new int64_t[traces.size()];
        int64_t *alignments_b = new int64_t[detections.size()];

        solve(traces.size(), detections.size(), cost_mtx.data(), false, alignments_a, alignments_b);
        EI_LOGD("detections size %zu\n", detections.size());
        EI_LOGD("traces size %zu\n", traces.size());

        for (size_t i = 0; i < traces.size(); i++) {
            EI_LOGD("alignments_a[%zu] %lld\n", i, alignments_a[i]);
        }

        for (size_t i = 0; i < detections.size(); i++) {
            EI_LOGD("alignments_b[%zu] %lld\n", i, alignments_b[i]);
        }

        std::vector<std::tuple<int, int, float>> matches;

        for (size_t i = 0; i < traces.size(); i++) {
            size_t trace_idx = i;
            size_t detection_idx = alignments_b[alignments_a[i]];
            float iou = 1 - cost_mtx[trace_idx * detections.size() + detection_idx];
            if (iou > iou_threshold) {
                matches.emplace_back(trace_idx, detection_idx, iou);
            }
        }
        delete[] alignments_a;
        delete[] alignments_b;
        return matches;
    }

    float iou_threshold;
};

class GreedyAlignment {
public:
    GreedyAlignment(float iou_threshold) : iou_threshold(iou_threshold) {
    }
    std::vector<std::tuple<int, int, float>> align(const std::vector<ei_impulse_result_bounding_box_t> traces,
                                                   const std::vector<ei_impulse_result_bounding_box_t> detections) {

        if (traces.empty() || detections.empty()) {
            return {};
        }

        std::vector<std::tuple<int, int, float>> alignments;
        for (size_t trace_idx = 0; trace_idx < traces.size(); ++trace_idx) {
            for (size_t detection_idx = 0; detection_idx < detections.size(); ++detection_idx) {
                float iou = intersection_over_union(traces[trace_idx], detections[detection_idx]);
                float cost = 1 - iou;
                EI_LOGD("t_idx=%zu d_idx=%zu cost=%.6f\n", trace_idx, detection_idx, cost);
                if (iou > iou_threshold) {
                    alignments.emplace_back(trace_idx, detection_idx, cost);
                }
            }
        }

        std::sort(alignments.begin(), alignments.end(), compare_tuples);
        EI_LOGD("alignments.size() %zu\n", alignments.size());
        std::vector<std::tuple<int, int, float>> matches;
        std::set<int> trace_idxs_matched;
        std::set<int> detection_idxs_matched;

        for (size_t i = 0; i < alignments.size(); i++) {
            uint32_t trace_idx = std::get<0>(alignments[i]);
            uint32_t detection_idx = std::get<1>(alignments[i]);
            float cost = std::get<2>(alignments[i]);

            if (trace_idxs_matched.find(trace_idx) == trace_idxs_matched.end() && detection_idxs_matched.find(detection_idx) == detection_idxs_matched.end()) {
                // (1 - cost) to get iou
                matches.emplace_back(trace_idx, detection_idx, 1 - cost);
                trace_idxs_matched.insert(trace_idx);
                if (trace_idxs_matched.size() == traces.size()) return matches;
                detection_idxs_matched.insert(detection_idx);
                if (detection_idxs_matched.size() == detections.size()) return matches;
            }
        }

        return matches;
    }

    float iou_threshold;
};