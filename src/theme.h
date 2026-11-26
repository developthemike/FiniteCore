#pragma once

#include <cstdint>

namespace fc::theme {

// Exact RGBA8 colours shared by the renderer and the GIF encoder (which
// reserves palette entries for them so text and background never flicker).
constexpr uint8_t kBackground[3] = {19, 20, 24};
constexpr uint8_t kTitle[3] = {240, 241, 245};
constexpr uint8_t kText[3] = {190, 195, 208};
constexpr uint8_t kLabel[3] = {215, 218, 228};

}  // namespace fc::theme
