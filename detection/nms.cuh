#pragma once
struct Det { float cx, cy, w, h, score; int cls; };
int nms(const Det* cand, int n, float iou_thresh, Det* out);