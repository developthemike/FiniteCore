#pragma once

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace fc {

// RGBA8 pixels, row-major, top row first. Alpha is dropped.
bool writePNG(const std::string& path, const uint8_t* rgba, int width, int height);

// Animated GIF (looping forever). Every frame gets its own 256-colour palette
// from median-cut quantisation, so lit 3D surfaces and smooth colormaps both
// quantise well. A few exact colours (background, text) can be reserved so
// they stay identical from frame to frame.
class GifWriter {
public:
    using Color = std::array<uint8_t, 3>;

    GifWriter() = default;
    ~GifWriter();
    GifWriter(const GifWriter&) = delete;
    GifWriter& operator=(const GifWriter&) = delete;

    bool open(const std::string& path, int width, int height, double fps, const std::vector<Color>& reserved = {});
    void addFrame(const uint8_t* rgba);
    void close();
    bool isOpen() const { return file_ != nullptr; }
    int frames() const { return frames_; }

private:
    void quantize(const uint8_t* rgba, std::vector<uint8_t>& indices);
    void writeImageData(const std::vector<uint8_t>& indices);

    std::FILE* file_ = nullptr;
    int width_ = 0, height_ = 0, delayCs_ = 5, frames_ = 0;
    std::vector<Color> reserved_;
    uint8_t palette_[256 * 3] = {};
    // Median-cut working storage (6 bits per channel histogram).
    std::vector<uint32_t> histCount_;
    std::vector<uint64_t> histSum_;    // 3 sums per bin
    std::vector<uint8_t> binIndex_;    // bin -> palette index
    // LZW dictionary: (prefix << 8 | byte) -> code, with generation stamps
    // so that a dictionary reset costs nothing.
    std::vector<uint16_t> dictCode_;
    std::vector<uint32_t> dictStamp_;
    uint32_t generation_ = 0;
};

}  // namespace fc
