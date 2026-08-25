#ifndef VPIPE_U15_IMAGE_GOLDEN_H
#define VPIPE_U15_IMAGE_GOLDEN_H

// A 24x24 luminance thumbnail of a KNOWN-GOOD render, and the only
// check in this repo that can tell a picture from a textured field.
//
// Why it exists. Every other check here is a non-degeneracy proxy --
// not a constant, sd > 25, more steps differ, the prompt changes it --
// and a checkpoint whose modulation weights were being read at the
// wrong dtype passed all four while rendering tiles. Twelve of twelve
// green, no image. A reference the shape of the answer is the only
// thing that catches that class.
//
// Why a 24x24 THUMBNAIL and a correlation rather than pixels. Two
// correct runs are not bit-identical -- swapping the matrix-core
// kernels for the simdgroup ones perturbs each step and diffusion
// carries the difference forward -- so a pixel bar would fail on a
// change that is fine. MEASURED on an M5, this model, 384x384 / 8
// steps:
//
//   two good renders, matmul2d vs steel      0.9898
//   good vs the wrong-dtype render           0.2109
//
// so the 0.90 bar below clears a whole kernel-path swap and still
// rejects the failure by a factor of four.
//
// PINNED to the generator defaults it was made with (384x384, 8 steps,
// seed 42, the default prompt). The check skips itself when any of
// those is overridden, because the golden would then describe a
// different image.
namespace u15_golden {

inline constexpr int kSize   = 24;
inline constexpr int kWidth  = 384;
inline constexpr int kHeight = 384;
inline constexpr int kSteps  = 8;
inline constexpr unsigned long long kSeed = 42;
inline constexpr double kMinCorrelation = 0.90;

// Row-major, 24x24, 0..255 luminance.
inline constexpr unsigned char kThumb[kSize * kSize] = {
    254, 255, 255, 255, 255, 255, 254, 254, 254, 254, 255, 255, 255, 255, 254, 254,
    254, 255, 254, 254, 254, 253, 254, 253, 254, 255, 255, 255, 255, 255, 255, 255,
    254, 254, 254, 255, 255, 255, 254, 254, 255, 255, 255, 254, 254, 254, 254, 254,
    254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 254, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 254, 254, 254, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 254, 255, 255, 255, 254, 255, 255, 255, 255, 255, 254, 254, 254, 254,
    254, 255, 255, 255, 255, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 254, 254, 254, 254, 254, 255, 255, 255, 255, 254, 254, 254,
    255, 222, 218, 254, 255, 255, 254, 241, 237, 255, 255, 255, 254, 254, 254, 254,
    254, 255, 254, 255, 255, 254, 253, 254, 254, 206, 160, 171, 246, 252, 211, 161,
    192, 255, 255, 255, 255, 254, 254, 254, 253, 254, 254, 254, 254, 254, 253, 254,
    254, 214, 146, 135, 130, 156, 162,  95, 198, 254, 255, 255, 255, 254, 254, 255,
    254, 254, 254, 254, 253, 253, 252, 253, 254, 220, 131, 134, 118, 123, 115, 103,
    207, 254, 255, 255, 254, 254, 254, 255, 253, 254, 253, 253, 253, 252, 252, 252,
    252, 219, 154, 133, 124, 129, 145, 132, 216, 254, 254, 255, 254, 254, 254, 255,
    254, 254, 253, 253, 252, 251, 252, 251, 249, 202,  98, 103,  84,  75, 114, 150,
    216, 253, 254, 254, 254, 254, 254, 255, 254, 254, 253, 252, 252, 251, 252, 251,
    244, 197, 159, 139, 101, 131, 133, 153, 201, 250, 254, 254, 254, 254, 254, 254,
    254, 254, 253, 253, 253, 251, 252, 250, 227, 184, 141, 171, 156, 169, 176, 188,
    186, 231, 252, 253, 253, 253, 254, 254, 254, 254, 254, 253, 253, 252, 253, 251,
    209, 151, 124, 132, 164, 171, 145, 156, 163, 190, 239, 252, 253, 254, 254, 254,
    253, 254, 253, 254, 253, 253, 254, 247, 179, 143, 124, 130, 128, 126, 119, 137,
    147, 159, 189, 233, 251, 254, 254, 254, 253, 253, 253, 253, 253, 252, 245, 207,
    159, 131, 103,  92, 101,  91, 105, 115, 124, 140, 142, 173, 231, 254, 254, 254,
    252, 253, 252, 252, 252, 245, 205, 178, 140, 108,  54,  46,  62,  66,  72,  59,
     85, 106, 121, 141, 184, 244, 252, 253, 253, 252, 252, 251, 247, 224, 194, 153,
    119,  81,  24,  33,  20,  14,  23,  19,  48,  75, 105, 118, 150, 207, 244, 249,
    251, 248, 249, 248, 245, 236, 182, 123,  83,  50,  20,  25,  23,  13,  19,  45,
     30,  43,  66,  76, 121, 180, 232, 243, 249, 248, 249, 248, 247, 241, 220, 206,
    164, 131,  98, 116,  89,  50, 109, 201, 134, 220, 194, 166, 171, 221, 246, 249,
    249, 249, 249, 250, 249, 248, 246, 245, 245, 244, 243, 242, 244, 243, 238, 241,
    247, 249, 249, 249, 249, 249, 249, 250, 254, 254, 254, 254, 254, 254, 254, 254,
    254, 254, 254, 254, 255, 255, 255, 254, 254, 254, 254, 254, 254, 254, 254, 254,
    254, 254, 254, 254, 254, 255, 255, 255, 255, 255, 254, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 254, 255, 254, 255, 253, 254, 255, 255, 254, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
};

}  // namespace u15_golden

#endif  // VPIPE_U15_IMAGE_GOLDEN_H
