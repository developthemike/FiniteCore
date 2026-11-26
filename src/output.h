#pragma once

#include <condition_variable>
#include <deque>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "fem.h"
#include "image_io.h"
#include "mesh.h"
#include "problem.h"

namespace fc {

// Runs file-writing jobs in order on a background thread.
class AsyncWriter {
public:
    AsyncWriter();
    ~AsyncWriter();
    void push(std::function<void()> job);
    void finish();

private:
    void run();
    std::mutex mutex_;
    std::condition_variable cv_, space_;
    std::deque<std::function<void()>> jobs_;
    bool done_ = false;
    std::thread thread_;  // declared last: starts after the members above exist
};

struct FieldStats {
    double tmin = 0, tmax = 0, tmean = 0;
    bool finite = true;
};

FieldStats computeStats(const std::vector<float>& T, const std::vector<double>& weights);

// Writes the time series: CSV (stats, probes, errors), VTK (.vtu + .pvd for
// ParaView), PNG frames and an animated GIF.
class OutputManager {
public:
    OutputManager(const ProblemDef& p, const Mesh& mesh, const FemSystem& sys);
    ~OutputManager();

    bool wantsImages() const { return png_ || gif_; }
    bool enabled() const { return enabled_; }
    void disable() { enabled_ = false; }
    const std::string& directory() const { return dir_; }

    // T: nodal temperatures; rgba: rendered frame (may be null if !wantsImages()).
    void write(double t, int step, const std::vector<float>& T, const std::vector<uint8_t>* rgba, uint32_t width,
               uint32_t height);
    void finish();

    int frames() const { return frame_; }
    std::vector<std::string> writtenFiles() const;

    // Verification against [verify] exact.
    bool hasExact() const { return hasExact_; }
    double lastErrorMax() const { return lastErrMax_; }
    double lastErrorL2() const { return lastErrL2_; }
    double worstErrorMax() const { return worstErrMax_; }

private:
    struct Probe {
        std::string name;
        uint32_t n[4];  // unused entries have weight 0
        double w[4];
    };

    const Mesh& mesh_;
    const FemSystem& sys_;
    std::string dir_;
    bool enabled_ = true;
    bool png_, gif_, vtk_, csv_;
    double gifFps_;
    std::vector<Probe> probes_;
    bool hasExact_ = false;
    Expr exact_;
    double lastErrMax_ = 0, lastErrL2_ = 0, worstErrMax_ = 0;

    std::ofstream csvFile_;
    GifWriter gifWriter_;
    std::vector<std::pair<double, std::string>> pvd_;
    int frame_ = 0;
    AsyncWriter writer_;
};

}  // namespace fc
