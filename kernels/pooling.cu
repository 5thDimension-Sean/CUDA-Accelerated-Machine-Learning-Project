// ============================================================================
// pooling.cu — Week 5
// Goal: Implement 2D max pooling in CUDA.
//
// Max pooling: slide a window over the input, output the maximum value.
//   Input:  [H x W x C]
//   Output: [H/stride x W/stride x C]
//
// Backward pass: gradient flows only through the element that was the max.
// You need to save which element WAS the max during forward (called a "mask").
//
// Also implement: average pooling (output = mean of window, not max).
// ============================================================================

// TODO — Week 5: implement here
