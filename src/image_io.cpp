#include "image_io.h"

#include <zlib.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>


namespace fc {

// ---- PNG ----------------------------------------------------------------------

namespace {

void put32be(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back(static_cast<uint8_t>(x >> 24));
    v.push_back(static_cast<uint8_t>(x >> 16));
    v.push_back(static_cast<uint8_t>(x >> 8));
    v.push_back(static_cast<uint8_t>(x));
}

void writeChunk(std::FILE* f, const char* type, const uint8_t* data, size_t len) {
    std::vector<uint8_t> head;
    put32be(head, static_cast<uint32_t>(len));
    std::fwrite(head.data(), 1, 4, f);
    std::fwrite(type, 1, 4, f);
    if (len) std::fwrite(data, 1, len, f);
    uLong crc = crc32(0L, Z_NULL, 0);
    crc = crc32(crc, reinterpret_cast<const Bytef*>(type), 4);
    if (len) crc = crc32(crc, data, static_cast<uInt>(len));
    std::vector<uint8_t> tail;
    put32be(tail, static_cast<uint32_t>(crc));
    std::fwrite(tail.data(), 1, 4, f);
}

}  // namespace

bool writePNG(const std::string& path, const uint8_t* rgba, int width, int height) {
    const size_t stride = static_cast<size_t>(width) * 3 + 1;
    std::vector<uint8_t> raw(stride * height);
    for (int y = 0; y < height; ++y) {
        uint8_t* row = raw.data() + stride * y;
        const uint8_t* src = rgba + static_cast<size_t>(y) * width * 4;
        row[0] = 1;  // "Sub" filter: smooth colormap images compress much better
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < 3; ++c) {
                const uint8_t cur = src[4 * x + c];
                const uint8_t left = x > 0 ? src[4 * (x - 1) + c] : 0;
                row[1 + 3 * x + c] = static_cast<uint8_t>(cur - left);
            }
        }
    }
    uLongf zlen = compressBound(static_cast<uLong>(raw.size()));
    std::vector<uint8_t> z(zlen);
    if (compress2(z.data(), &zlen, raw.data(), static_cast<uLong>(raw.size()), 6) != Z_OK) return false;

    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    std::fwrite(sig, 1, 8, f);
    std::vector<uint8_t> ihdr;
    put32be(ihdr, static_cast<uint32_t>(width));
    put32be(ihdr, static_cast<uint32_t>(height));
    ihdr.insert(ihdr.end(), {8, 2, 0, 0, 0});  // 8-bit RGB, deflate, adaptive filtering, no interlace
    writeChunk(f, "IHDR", ihdr.data(), ihdr.size());
    writeChunk(f, "IDAT", z.data(), zlen);
    writeChunk(f, "IEND", nullptr, 0);
    const bool ok = std::ferror(f) == 0;
    std::fclose(f);
    return ok;
}

// ---- GIF ----------------------------------------------------------------------

namespace {

// Packs LZW codes LSB-first into 255-byte GIF sub-blocks.
class GifBitWriter {
public:
    explicit GifBitWriter(std::FILE* f) : f_(f) {}
    void put(uint32_t code, int bits) {
        acc_ |= static_cast<uint64_t>(code) << nbits_;
        nbits_ += bits;
        while (nbits_ >= 8) {
            byte(static_cast<uint8_t>(acc_ & 0xff));
            acc_ >>= 8;
            nbits_ -= 8;
        }
    }
    void finish() {
        if (nbits_ > 0) byte(static_cast<uint8_t>(acc_ & 0xff));
        acc_ = 0;
        nbits_ = 0;
        flushBlock();
        std::fputc(0, f_);  // block terminator
    }

private:
    void byte(uint8_t b) {
        block_[len_++] = b;
        if (len_ == 255) flushBlock();
    }
    void flushBlock() {
        if (len_ == 0) return;
        std::fputc(len_, f_);
        std::fwrite(block_, 1, len_, f_);
        len_ = 0;
    }
    std::FILE* f_;
    uint64_t acc_ = 0;
    int nbits_ = 0;
    uint8_t block_[255];
    int len_ = 0;
};

void put16le(std::FILE* f, int v) {
    std::fputc(v & 0xff, f);
    std::fputc((v >> 8) & 0xff, f);
}

constexpr int kBinBits = 6;  // histogram resolution per channel
constexpr uint32_t kBins = 1u << (3 * kBinBits);

inline uint32_t binOf(const uint8_t* p) {
    constexpr int s = 8 - kBinBits;
    return (static_cast<uint32_t>(p[0] >> s) << (2 * kBinBits)) | (static_cast<uint32_t>(p[1] >> s) << kBinBits) |
           static_cast<uint32_t>(p[2] >> s);
}

}  // namespace

GifWriter::~GifWriter() { close(); }

bool GifWriter::open(const std::string& path, int width, int height, double fps, const std::vector<Color>& reserved) {
    close();
    file_ = std::fopen(path.c_str(), "wb");
    if (!file_) return false;
    width_ = width;
    height_ = height;
    delayCs_ = std::max(2, static_cast<int>(std::lround(100.0 / fps)));
    frames_ = 0;
    reserved_.assign(reserved.begin(), reserved.begin() + std::min<size_t>(reserved.size(), 16));
    histCount_.assign(kBins, 0);
    histSum_.assign(static_cast<size_t>(kBins) * 3, 0);
    binIndex_.assign(kBins, 0);
    dictCode_.assign(1u << 20, 0);
    dictStamp_.assign(1u << 20, 0);
    generation_ = 0;

    std::fwrite("GIF89a", 1, 6, file_);
    put16le(file_, width);
    put16le(file_, height);
    std::fputc(0x70, file_);  // no global colour table (every frame has its own), 8 bits/colour
    std::fputc(0, file_);     // background index
    std::fputc(0, file_);     // pixel aspect ratio
    // NETSCAPE2.0 application extension: loop forever
    static const uint8_t loop[] = {0x21, 0xFF, 0x0B, 'N', 'E', 'T', 'S', 'C', 'A', 'P', 'E', '2', '.', '0',
                                   0x03, 0x01, 0x00, 0x00, 0x00};
    std::fwrite(loop, 1, sizeof(loop), file_);
    return true;
}

// Median cut over a 6-bit-per-channel histogram. Reserved colours get the first
// palette entries and are matched exactly; everything else is split into boxes
// (largest population x extent first) whose weighted means form the palette.
void GifWriter::quantize(const uint8_t* rgba, std::vector<uint8_t>& idx) {
    const size_t count = static_cast<size_t>(width_) * height_;
    const int nReserved = static_cast<int>(reserved_.size());
    uint32_t reservedKey[16];
    for (int r = 0; r < nReserved; ++r)
        reservedKey[r] = (uint32_t(reserved_[r][0]) << 16) | (uint32_t(reserved_[r][1]) << 8) | reserved_[r][2];

    std::fill(histCount_.begin(), histCount_.end(), 0u);
    std::fill(histSum_.begin(), histSum_.end(), 0ull);
    idx.assign(count, 0);
    std::vector<uint8_t> isReserved(count, 0);
    for (size_t i = 0; i < count; ++i) {
        const uint8_t* p = rgba + 4 * i;
        const uint32_t key = (uint32_t(p[0]) << 16) | (uint32_t(p[1]) << 8) | p[2];
        int r = 0;
        while (r < nReserved && reservedKey[r] != key) ++r;
        if (r < nReserved) {
            idx[i] = static_cast<uint8_t>(r);
            isReserved[i] = 1;
            continue;
        }
        const uint32_t b = binOf(p);
        histCount_[b]++;
        histSum_[3 * b] += p[0];
        histSum_[3 * b + 1] += p[1];
        histSum_[3 * b + 2] += p[2];
    }

    struct Entry {
        uint32_t bin, count;
        uint8_t c[3];  // bin coordinates per channel
    };
    std::vector<Entry> entries;
    for (uint32_t b = 0; b < kBins; ++b)
        if (histCount_[b])
            entries.push_back({b, histCount_[b],
                               {static_cast<uint8_t>(b >> (2 * kBinBits)), static_cast<uint8_t>((b >> kBinBits) & 63),
                                static_cast<uint8_t>(b & 63)}});

    struct Box {
        size_t begin, end;
        uint64_t count;
        int axis, extent;
    };
    auto measure = [&entries](Box& box) {
        int lo[3] = {255, 255, 255}, hi[3] = {0, 0, 0};
        box.count = 0;
        for (size_t k = box.begin; k < box.end; ++k) {
            box.count += entries[k].count;
            for (int a = 0; a < 3; ++a) {
                lo[a] = std::min<int>(lo[a], entries[k].c[a]);
                hi[a] = std::max<int>(hi[a], entries[k].c[a]);
            }
        }
        box.axis = 0;
        box.extent = -1;
        for (int a = 0; a < 3; ++a)
            if (hi[a] - lo[a] > box.extent) {
                box.extent = hi[a] - lo[a];
                box.axis = a;
            }
    };
    std::vector<Box> boxes;
    if (!entries.empty()) {
        boxes.push_back({0, entries.size(), 0, 0, 0});
        measure(boxes[0]);
    }
    const size_t maxBoxes = static_cast<size_t>(256 - nReserved);
    while (boxes.size() < maxBoxes) {
        size_t best = boxes.size();
        double bestScore = 0.0;
        for (size_t k = 0; k < boxes.size(); ++k) {
            const double score = static_cast<double>(boxes[k].count) * boxes[k].extent;
            if (boxes[k].end - boxes[k].begin > 1 && score > bestScore) {
                bestScore = score;
                best = k;
            }
        }
        if (best == boxes.size()) break;  // nothing left to split
        Box box = boxes[best];
        const int axis = box.axis;
        std::sort(entries.begin() + box.begin, entries.begin() + box.end,
                  [axis](const Entry& a, const Entry& b) { return a.c[axis] < b.c[axis]; });
        uint64_t acc = 0;
        size_t split = box.begin;
        while (split < box.end - 1 && acc + entries[split].count <= box.count / 2) acc += entries[split++].count;
        if (split == box.begin) split = box.begin + 1;
        Box lo{box.begin, split, 0, 0, 0}, hi{split, box.end, 0, 0, 0};
        measure(lo);
        measure(hi);
        boxes[best] = lo;
        boxes.push_back(hi);
    }

    std::memset(palette_, 0, sizeof(palette_));
    for (int r = 0; r < nReserved; ++r)
        for (int ch = 0; ch < 3; ++ch) palette_[3 * r + ch] = reserved_[r][ch];
    for (size_t k = 0; k < boxes.size(); ++k) {
        const int pi = nReserved + static_cast<int>(k);
        uint64_t sum[3] = {0, 0, 0}, n = 0;
        for (size_t e = boxes[k].begin; e < boxes[k].end; ++e) {
            const uint32_t b = entries[e].bin;
            for (int ch = 0; ch < 3; ++ch) sum[ch] += histSum_[3 * b + ch];
            n += histCount_[b];
        }
        for (int ch = 0; ch < 3; ++ch) palette_[3 * pi + ch] = static_cast<uint8_t>((sum[ch] + n / 2) / std::max<uint64_t>(n, 1));
    }
    // Map every used bin (by its mean colour) to the nearest palette entry. Using
    // the box a bin was sorted into instead leaves stray pixels where a long box
    // averages over quite different colours.
    const int paletteSize = nReserved + static_cast<int>(boxes.size());
    for (const Entry& e : entries) {
        const uint32_t b = e.bin;
        const int64_t c[3] = {static_cast<int64_t>((histSum_[3 * b] + histCount_[b] / 2) / histCount_[b]),
                              static_cast<int64_t>((histSum_[3 * b + 1] + histCount_[b] / 2) / histCount_[b]),
                              static_cast<int64_t>((histSum_[3 * b + 2] + histCount_[b] / 2) / histCount_[b])};
        int best = nReserved;
        int64_t bestD = INT64_MAX;
        for (int pi = nReserved; pi < paletteSize; ++pi) {
            const int64_t dr = c[0] - palette_[3 * pi], dg = c[1] - palette_[3 * pi + 1], db = c[2] - palette_[3 * pi + 2];
            const int64_t d = 2 * dr * dr + 4 * dg * dg + 3 * db * db;
            if (d < bestD) {
                bestD = d;
                best = pi;
            }
        }
        binIndex_[b] = static_cast<uint8_t>(best);
    }
    for (size_t i = 0; i < count; ++i)
        if (!isReserved[i]) idx[i] = binIndex_[binOf(rgba + 4 * i)];
}

void GifWriter::addFrame(const uint8_t* rgba) {
    if (!file_) return;
    std::vector<uint8_t> idx;
    quantize(rgba, idx);
    // Graphic control extension: frame delay
    std::fputc(0x21, file_);
    std::fputc(0xF9, file_);
    std::fputc(0x04, file_);
    std::fputc(0x04, file_);  // disposal: do not dispose
    put16le(file_, delayCs_);
    std::fputc(0, file_);
    std::fputc(0, file_);
    // Image descriptor (full frame) with a local 256-entry colour table
    std::fputc(0x2C, file_);
    put16le(file_, 0);
    put16le(file_, 0);
    put16le(file_, width_);
    put16le(file_, height_);
    std::fputc(0x87, file_);
    std::fwrite(palette_, 1, sizeof(palette_), file_);
    writeImageData(idx);
    ++frames_;
}

void GifWriter::writeImageData(const std::vector<uint8_t>& idx) {
    constexpr int kMinCodeSize = 8;
    constexpr uint32_t kClear = 256, kEnd = 257;
    std::fputc(kMinCodeSize, file_);
    GifBitWriter bw(file_);

    int codeSize = kMinCodeSize + 1;
    uint32_t next = kEnd + 1;
    ++generation_;
    bw.put(kClear, codeSize);
    uint32_t prefix = idx[0];
    for (size_t i = 1; i < idx.size(); ++i) {
        const uint32_t k = idx[i];
        const uint32_t key = (prefix << 8) | k;
        if (dictStamp_[key] == generation_) {
            prefix = dictCode_[key];
            continue;
        }
        bw.put(prefix, codeSize);
        if (next < 4096) {
            dictCode_[key] = static_cast<uint16_t>(next);
            dictStamp_[key] = generation_;
            ++next;
            // The decoder lags one entry behind; it widens codes once the
            // table holds more entries than the current width can address.
            if (next > (1u << codeSize) && codeSize < 12) ++codeSize;
        } else {
            bw.put(kClear, codeSize);
            ++generation_;
            codeSize = kMinCodeSize + 1;
            next = kEnd + 1;
        }
        prefix = k;
    }
    bw.put(prefix, codeSize);
    // After reading the last code the decoder adds one more table entry,
    // which may widen the end-of-information code.
    if (next < 4096 && next + 1 > (1u << codeSize) && codeSize < 12) ++codeSize;
    bw.put(kEnd, codeSize);
    bw.finish();
}

void GifWriter::close() {
    if (!file_) return;
    std::fputc(0x3B, file_);  // trailer
    std::fclose(file_);
    file_ = nullptr;
}

}  // namespace fc
