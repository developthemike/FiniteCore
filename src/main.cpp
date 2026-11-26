// FiniteCore: transient heat conduction with linear finite elements, time-stepped
// on the GPU with Vulkan compute and animated live with Vulkan graphics.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "config.h"
#include "cpu_solver.h"
#include "fem.h"
#include "gpu_solver.h"
#include "image_io.h"
#include "mesh.h"
#include "output.h"
#include "presenter.h"
#include "problem.h"
#include "renderer.h"
#include "simulation.h"
#include "vk_context.h"

using namespace fc;

namespace {

using Clock = std::chrono::steady_clock;
double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

struct Options {
    std::string config;
    std::vector<std::string> overrides;
    bool headless = false;
    int device = -1;
    bool listDevices = false;
    bool compareCpu = false;
    bool exitWhenDone = false;
    bool validation = false;
    bool noOutput = false;
    bool quiet = false;
    bool profile = false;
};

void printUsage() {
    std::printf(
        "usage: finitecore <config.ini> [options]\n"
        "\n"
        "  --headless          run without a window (file output only)\n"
        "  --set S.K=V         override a config value, e.g. --set mesh.nx=128 --set time.dt=0.02\n"
        "  --no-output         do not write any files\n"
        "  --compare-cpu       also run a double-precision CPU reference and report the difference\n"
        "  --device N          use Vulkan device N (see --list-devices)\n"
        "  --list-devices      list Vulkan devices and exit\n"
        "  --exit-when-done    close the window when the simulation has finished\n"
        "  --validation        enable the Vulkan validation layer (if installed)\n"
        "  --quiet             no progress output\n"
        "  --profile           print GPU submission statistics\n"
        "\n"
        "window keys: space pause | r restart | +/- speed | m mesh | c colormap | s screenshot | q quit\n");
}

Options parseArgs(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string(name) + " needs a value");
            return argv[++i];
        };
        if (a == "-h" || a == "--help") {
            printUsage();
            std::exit(0);
        } else if (a == "--headless") {
            o.headless = true;
        } else if (a == "--set") {
            o.overrides.push_back(value("--set"));
        } else if (a.rfind("--set=", 0) == 0) {
            o.overrides.push_back(a.substr(6));
        } else if (a == "--device") {
            o.device = std::stoi(value("--device"));
        } else if (a == "--list-devices") {
            o.listDevices = true;
        } else if (a == "--compare-cpu") {
            o.compareCpu = true;
        } else if (a == "--exit-when-done") {
            o.exitWhenDone = true;
        } else if (a == "--validation") {
            o.validation = true;
        } else if (a == "--no-output") {
            o.noOutput = true;
        } else if (a == "--quiet") {
            o.quiet = true;
        } else if (a == "--profile") {
            o.profile = true;
        } else if (!a.empty() && a[0] == '-') {
            throw std::runtime_error("unknown option '" + a + "' (see --help)");
        } else if (o.config.empty()) {
            o.config = a;
        } else {
            throw std::runtime_error("unexpected argument '" + a + "'");
        }
    }
    if (o.config.empty() && !o.listDevices) {
        printUsage();
        std::exit(1);
    }
    return o;
}

// Colour scale: fixed, per-frame ("dynamic"), or "auto" (grows to cover every value seen).
struct ColorRange {
    std::string mode = "auto";
    float lo = 0, hi = 1;
    void init(const ViewDef& v, const std::vector<double>& T0) {
        mode = v.range;
        if (mode == "fixed") {
            lo = static_cast<float>(v.rangeMin);
            hi = static_cast<float>(v.rangeMax);
            return;
        }
        double a = *std::min_element(T0.begin(), T0.end()), b = *std::max_element(T0.begin(), T0.end());
        if (b - a < 1e-9 * std::max(1.0, std::fabs(a))) {
            a -= 1.0;
            b += 1.0;
        }
        lo = static_cast<float>(a);
        hi = static_cast<float>(b);
    }
    void update(const FieldStats& st) {
        if (mode == "fixed" || !st.finite) return;
        if (mode == "dynamic") {
            lo = static_cast<float>(st.tmin);
            hi = static_cast<float>(st.tmax);
            if (hi - lo < 1e-6f * std::max(1.0f, std::fabs(lo))) {
                lo -= 0.5f;
                hi += 0.5f;
            }
            return;
        }
        lo = std::min(lo, static_cast<float>(st.tmin));
        hi = std::max(hi, static_cast<float>(st.tmax));
    }
};

std::string formatTime(double t, double dt) {
    const int decimals = std::clamp(static_cast<int>(std::ceil(-std::log10(dt) - 1e-9)), 0, 6);
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*f", decimals, t);
    return buf;
}

// 3D camera/cut settings from the config (cut point defaults to the domain centre).
View3D initialView(const ProblemDef& p, const Mesh& m) {
    View3D v;
    v.mode = p.view.mode;
    v.axis = p.view.axis;
    v.yaw = p.view.yaw;
    v.pitch = p.view.pitch;
    v.zoom = p.view.zoom;
    const double center[3] = {0.5 * (m.xmin + m.xmax), 0.5 * (m.ymin + m.ymax), 0.5 * (m.zmin + m.zmax)};
    for (int a = 0; a < 3; ++a) v.cut[a] = p.view.cutSet ? p.view.cut[a] : center[a];
    return v;
}

// Camera turn from [view] spin (degrees per second of playback) at simulated time t.
double spinAngle(const ProblemDef& p, const FemSystem& sys, double t) {
    return p.view.spin * (t / sys.tEnd) * p.view.playback;
}

FrameInfo makeFrame(const ProblemDef& p, const Simulation& sim, const FemSystem& sys, const ColorRange& range,
                    const FieldStats& st, int colormap, bool mesh, const char* tag, const View3D& view) {
    FrameInfo f;
    f.title = p.title;
    char buf[256];
    std::snprintf(buf, sizeof(buf), "t = %s s   step %d / %d%s", formatTime(sim.time(), sys.dt).c_str(), sim.step(),
                  sys.nSteps, tag);
    f.lines.push_back(buf);
    std::string cutInfo;
    if (p.mesh.dim == 3) {
        cutInfo = std::string("   view: ") + viewModeName(view.mode);
        if (view.mode == View3D::Clip) {
            char cbuf[64];
            std::snprintf(cbuf, sizeof(cbuf), " %c = %.4g", 'x' + view.axis, view.cut[view.axis]);
            cutInfo += cbuf;
        }
    }
    std::snprintf(buf, sizeof(buf), "min %.4g   max %.4g   mean %.4g %s%s", st.tmin, st.tmax, st.tmean,
                  p.view.unit.c_str(), cutInfo.c_str());
    f.lines.push_back(buf);
    f.unit = "T [" + p.view.unit + "]";
    f.rangeMin = range.lo;
    f.rangeMax = range.hi;
    f.colormap = colormap;
    f.contours = p.view.contours;
    f.showMesh = mesh;
    f.view = view;
    f.view.yaw += spinAngle(p, sys, sim.time());
    return f;
}

#ifdef FC_HAVE_GLFW
struct WindowInput {
    bool paused = false, reset = false, toggleMesh = false, screenshot = false;
    int colormapStep = 0;
    double speed = 1.0;
    // 3D navigation
    bool dragging = false, resetView = false;
    double lastX = 0, lastY = 0, dYaw = 0, dPitch = 0, zoomSteps = 0;
    int modeStep = 0, axis = -1, cutSteps = 0;
};

void mouseButtonCallback(GLFWwindow* w, int button, int action, int) {
    auto* in = static_cast<WindowInput*>(glfwGetWindowUserPointer(w));
    if (button != GLFW_MOUSE_BUTTON_LEFT) return;
    in->dragging = action == GLFW_PRESS;
    glfwGetCursorPos(w, &in->lastX, &in->lastY);
}

void cursorCallback(GLFWwindow* w, double x, double y) {
    auto* in = static_cast<WindowInput*>(glfwGetWindowUserPointer(w));
    if (!in->dragging) return;
    in->dYaw -= 0.35 * (x - in->lastX);
    in->dPitch -= 0.35 * (y - in->lastY);
    in->lastX = x;
    in->lastY = y;
}

void scrollCallback(GLFWwindow* w, double, double dy) {
    static_cast<WindowInput*>(glfwGetWindowUserPointer(w))->zoomSteps += dy;
}

void keyCallback(GLFWwindow* w, int key, int, int action, int) {
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
    auto* in = static_cast<WindowInput*>(glfwGetWindowUserPointer(w));
    switch (key) {
        case GLFW_KEY_SPACE: in->paused = !in->paused; break;
        case GLFW_KEY_R: in->reset = true; break;
        case GLFW_KEY_M: in->toggleMesh = true; break;
        case GLFW_KEY_C: in->colormapStep++; break;
        case GLFW_KEY_S: in->screenshot = true; break;
        case GLFW_KEY_EQUAL:
        case GLFW_KEY_KP_ADD:
        case GLFW_KEY_RIGHT_BRACKET: in->speed = std::min(in->speed * 2.0, 1024.0); break;
        case GLFW_KEY_MINUS:
        case GLFW_KEY_KP_SUBTRACT:
        case GLFW_KEY_LEFT_BRACKET: in->speed = std::max(in->speed / 2.0, 1.0 / 1024.0); break;
        case GLFW_KEY_V: in->modeStep++; break;
        case GLFW_KEY_X: in->axis = 0; break;
        case GLFW_KEY_Y: in->axis = 1; break;
        case GLFW_KEY_Z: in->axis = 2; break;
        case GLFW_KEY_RIGHT:
        case GLFW_KEY_UP:
        case GLFW_KEY_PAGE_UP: in->cutSteps++; break;
        case GLFW_KEY_LEFT:
        case GLFW_KEY_DOWN:
        case GLFW_KEY_PAGE_DOWN: in->cutSteps--; break;
        case GLFW_KEY_HOME: in->resetView = true; break;
        case GLFW_KEY_ESCAPE:
        case GLFW_KEY_Q: glfwSetWindowShouldClose(w, GLFW_TRUE); break;
        default: break;
    }
}
#endif

DiagonalPattern patternFromName(const std::string& s) {
    if (s == "right") return DiagonalPattern::Right;
    if (s == "left") return DiagonalPattern::Left;
    return DiagonalPattern::Alternating;
}

int run(int argc, char** argv) {
    Options opt = parseArgs(argc, argv);
    if (opt.listDevices) {
        VulkanContext::listDevices();
        return 0;
    }

    Config cfg = Config::load(opt.config);
    for (const std::string& s : opt.overrides) cfg.applyOverride(s);
    ProblemDef prob = parseProblem(cfg);
    cfg.warnUnused();
    if (opt.noOutput) prob.output.png = prob.output.gif = prob.output.vtk = prob.output.csv = false;

    const auto tSetup = Clock::now();
    const MeshDef& md = prob.mesh;
    Mesh mesh;
    if (md.kind == "tube") {
        TubeSpec spec;
        std::copy(md.origin, md.origin + 3, spec.origin);
        spec.length = md.length;
        spec.bend = md.bend;
        spec.radius = [&md](double s) { return md.radius.eval(s, 0, 0, 0); };
        spec.section = [&md](double a, double s) { return md.section.eval(a, s, 0, 0); };
        if (md.cavity.valid()) spec.cavity = [&md](double s) { return md.cavity.eval(s, 0, 0, 0); };
        spec.cellsAlong = md.cells[0];
        spec.cellsAcross = md.cells[1];
        mesh = makeTubeMesh(spec);
    } else if (md.dim == 3) {
        mesh = makeBoxMesh(md.origin[0], md.origin[1], md.origin[2], md.size[0], md.size[1], md.size[2], md.cells[0],
                           md.cells[1], md.cells[2]);
    } else {
        mesh = makeRectangleMesh(md.origin[0], md.origin[1], md.size[0], md.size[1], md.cells[0], md.cells[1],
                                 patternFromName(md.pattern));
    }
    // The plot area of the image: the domain's aspect in 2D, a fixed 4:3-ish frame for the 3D camera.
    const double plotAspect = md.dim == 3 ? 1.3 : md.size[0] / md.size[1];
    Assembly assembly = assemble(prob, mesh);
    const FemSystem& sys = assembly.system;
    const double assembleSeconds = since(tSetup);

    const char* schemeName = sys.theta == 0.0 ? "explicit Euler" : sys.theta == 1.0 ? "backward Euler"
                             : sys.theta == 0.5                ? "Crank-Nicolson"
                                                               : "theta scheme";
    std::printf("FiniteCore - transient heat conduction, P1 finite elements, Vulkan compute\n");
    std::printf("  config    : %s  \"%s\"\n", opt.config.c_str(), prob.title.c_str());
    std::printf("  mesh      : %s -> %zu nodes, %zu %s, %zu nonzeros (assembled in %.0f ms)\n",
                mesh.description.c_str(), mesh.numNodes(), mesh.numCells(), mesh.dim == 3 ? "tetrahedra" : "triangles",
                sys.nnz(), assembleSeconds * 1e3);
    for (const std::string& n : assembly.notes) std::printf("  %s\n", n.c_str());
    std::printf("  time      : %s (theta=%g), %s mass, dt=%g s, %d steps to t=%g s\n", schemeName, sys.theta,
                sys.lumped ? "lumped" : "consistent", sys.dt, sys.nSteps, sys.tEnd);
    if (sys.stableDt > 0) std::printf("  stability : explicit bound dt <= %.4g s\n", sys.stableDt);
    if (!sys.diagonal)
        std::printf("  solver    : Jacobi-preconditioned CG on the GPU, tol=%g, max %d iterations\n",
                    prob.solver.tolerance, prob.solver.maxIterations);
    for (const std::string& w : assembly.warnings) std::fprintf(stderr, "warning: %s\n", w.c_str());

    // ---- window (optional) ----------------------------------------------------------
    VulkanContext::Options vopt;
    vopt.deviceIndex = opt.device;
    vopt.validation = opt.validation;
    bool wantWindow = prob.view.window && !opt.headless;
#ifdef FC_HAVE_GLFW
    GLFWwindow* window = nullptr;
    if (wantWindow) {
        if (!glfwInit()) {
            std::fprintf(stderr, "note: no display available, running headless\n");
            wantWindow = false;
        } else if (!glfwVulkanSupported()) {
            std::fprintf(stderr, "note: GLFW reports no Vulkan presentation support, running headless\n");
            glfwTerminate();
            wantWindow = false;
        }
    }
    if (wantWindow) {
        uint32_t count = 0;
        const char** exts = glfwGetRequiredInstanceExtensions(&count);
        for (uint32_t i = 0; i < count; ++i) vopt.instanceExtensions.push_back(exts[i]);
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        const int h = prob.view.height;
        const int w = prob.view.width > 0 ? prob.view.width
                                          : static_cast<int>(Renderer::preferredWidth(h, plotAspect));
        window = glfwCreateWindow(w, h, ("FiniteCore - " + prob.title).c_str(), nullptr, nullptr);
        if (!window) {
            std::fprintf(stderr, "note: could not create a window, running headless\n");
            glfwTerminate();
            wantWindow = false;
        } else {
            vopt.createSurface = [window](VkInstance inst) {
                VkSurfaceKHR s = VK_NULL_HANDLE;
                VK_CHECK(glfwCreateWindowSurface(inst, window, nullptr, &s));
                return s;
            };
        }
    }
#else
    if (wantWindow) std::fprintf(stderr, "note: built without GLFW, running headless\n");
    wantWindow = false;
#endif

    int exitCode = 0;
    {
        VulkanContext ctx(vopt);
        std::printf("  device    : %s (%s)\n", ctx.deviceName().c_str(), ctx.deviceTypeName().c_str());

        GpuSolver gpu(ctx, sys, prob.solver.tolerance, prob.solver.maxIterations);
        std::unique_ptr<CpuSolver> reference;
        if (opt.compareCpu) reference = std::make_unique<CpuSolver>(sys, prob.solver.tolerance, prob.solver.maxIterations);
        Renderer renderer(ctx, mesh, gpu.temperatureBuffer());
        OutputManager outputs(prob, mesh, sys);
        Simulation sim(sys, assembly.loads, gpu, reference.get(), prob.output.interval);

        RenderTarget outTarget;
        uint32_t outW = 0, outH = 0;
        if (outputs.enabled() && outputs.wantsImages()) {
            outH = static_cast<uint32_t>(prob.output.height);
            outW = prob.output.width > 0 ? static_cast<uint32_t>(prob.output.width)
                                         : Renderer::preferredWidth(outH, plotAspect);
            outTarget = renderer.createTarget(outW, outH, true);
        }

        ColorRange range;
        range.init(prob.view, sys.T0);
        const View3D configView = initialView(prob, mesh);
        std::vector<float> Th(sys.T0.begin(), sys.T0.end());
        std::vector<uint8_t> rgba;
        FieldStats stats = computeStats(Th, sys.lumpedMass);
        double maxRefDiff = 0.0;
        // Anything a million times beyond the initial/boundary scale means the scheme diverged.
        double divergeLimit = std::max({1.0, std::fabs(stats.tmin), std::fabs(stats.tmax),
                                        std::fabs(static_cast<double>(range.lo)), std::fabs(static_cast<double>(range.hi))}) * 1e6;

        auto captureOutput = [&]() {
            gpu.download(Th);
            stats = computeStats(Th, sys.lumpedMass);
            if (!stats.finite || std::max(std::fabs(stats.tmin), std::fabs(stats.tmax)) > divergeLimit) {
                char buf[256];
                std::snprintf(buf, sizeof(buf), "the solution diverged at t = %g s%s", sim.time(),
                              sys.theta < 0.5 ? " (explicit scheme unstable: reduce dt or use dt = auto)" : "");
                throw std::runtime_error(buf);
            }
            range.update(stats);
            if (reference) {
                const std::vector<double>& Tr = reference->temperature();
                for (size_t i = 0; i < Th.size(); ++i)
                    maxRefDiff = std::max(maxRefDiff, std::fabs(static_cast<double>(Th[i]) - Tr[i]));
            }
            const std::vector<uint8_t>* pixels = nullptr;
            if (outputs.enabled() && outputs.wantsImages()) {
                FrameInfo fi = makeFrame(prob, sim, sys, range, stats, prob.view.colormap, prob.view.mesh, "", configView);
                renderer.renderToMemory(outTarget, fi, rgba);
                pixels = &rgba;
            }
            outputs.write(sim.time(), sim.step(), Th, pixels, outW, outH);
            sim.markOutputWritten();
        };

        std::printf("  output    : %s\n", outputs.enabled() ? outputs.directory().c_str() : "(none)");
        std::printf("\n");
        std::fflush(stdout);
        const auto tRun = Clock::now();

#ifdef FC_HAVE_GLFW
        if (wantWindow) {
            std::printf("window: space pause | r restart | +/- speed | m mesh | c colormap | s screenshot | q quit\n");
            if (mesh.dim == 3)
                std::printf("3D:     drag rotate | scroll zoom | v view mode | x/y/z cut axis | arrows move cut | "
                            "home reset view\n");
            Presenter presenter(ctx, renderer, window);
            WindowInput input;
            glfwSetWindowUserPointer(window, &input);
            glfwSetKeyCallback(window, keyCallback);
            glfwSetMouseButtonCallback(window, mouseButtonCallback);
            glfwSetCursorPosCallback(window, cursorCallback);
            glfwSetScrollCallback(window, scrollCallback);
            View3D view = configView;
            const double lo[3] = {mesh.xmin, mesh.ymin, mesh.zmin}, hi[3] = {mesh.xmax, mesh.ymax, mesh.zmax};
            const double simRate = sys.tEnd / prob.view.playback;  // simulated seconds per wall second
            double simTarget = 0.0;
            bool showMesh = prob.view.mesh;
            int colormap = prob.view.colormap;
            auto lastFrame = Clock::now(), lastTitle = Clock::now();
            int stepsAtTitle = 0;
            double doneSince = -1.0;
            while (!glfwWindowShouldClose(window)) {
                glfwPollEvents();
                if (input.reset) {
                    input.reset = false;
                    sim.reset();
                    simTarget = 0.0;
                    stepsAtTitle = 0;
                    doneSince = -1.0;
                    range.init(prob.view, sys.T0);
                    if (outputs.enabled()) {
                        outputs.disable();
                        std::printf("restarted: file output stays as written by the first run\n");
                    }
                }
                if (input.toggleMesh) {
                    showMesh = !showMesh;
                    input.toggleMesh = false;
                }
                if (input.colormapStep) {
                    colormap = (colormap + input.colormapStep) % 4;
                    input.colormapStep = 0;
                }
                if (mesh.dim == 3) {
                    view.yaw += input.dYaw;
                    view.pitch = std::clamp(view.pitch + input.dPitch, -89.0, 89.0);
                    view.zoom = std::clamp(view.zoom * std::pow(1.15, input.zoomSteps), 0.2, 25.0);
                    input.dYaw = input.dPitch = input.zoomSteps = 0.0;
                    if (input.modeStep) {
                        view.mode = (view.mode + input.modeStep) % 4;
                        input.modeStep = 0;
                    }
                    if (input.axis >= 0) {  // choose the cut axis; from the plain surface view switch to a clip
                        view.axis = input.axis;
                        if (view.mode == View3D::Surface) view.mode = View3D::Clip;
                        input.axis = -1;
                    }
                    if (input.cutSteps) {
                        const int a = view.axis;
                        view.cut[a] = std::clamp(view.cut[a] + input.cutSteps * (hi[a] - lo[a]) / 40.0, lo[a], hi[a]);
                        input.cutSteps = 0;
                    }
                    if (input.resetView) {
                        view = configView;
                        input.resetView = false;
                    }
                } else {
                    input.dYaw = input.dPitch = input.zoomSteps = 0.0;
                    input.modeStep = input.cutSteps = 0;
                    input.axis = -1;
                }
                const auto now = Clock::now();
                const double frameDt = std::min(0.1, std::chrono::duration<double>(now - lastFrame).count());
                lastFrame = now;
                if (!input.paused && !sim.finished()) {
                    simTarget += frameDt * simRate * input.speed;
                    const auto budget = Clock::now();
                    while (!sim.finished()) {
                        if (sim.outputPending()) {
                            captureOutput();
                            continue;
                        }
                        const int want = static_cast<int>(std::floor((simTarget - sim.time()) / sys.dt + 1e-9));
                        if (want <= 0) break;
                        sim.advance(want);
                        if (since(budget) > 0.03) {  // can't keep up: drop the backlog, stay responsive
                            simTarget = sim.time();
                            break;
                        }
                    }
                }
                if (sim.outputPending() && sim.finished()) captureOutput();

                gpu.download(Th);
                stats = computeStats(Th, sys.lumpedMass);
                range.update(stats);
                const char* tag = input.paused ? "   [paused]" : sim.finished() ? "   [done]" : "";
                FrameInfo fi = makeFrame(prob, sim, sys, range, stats, colormap, showMesh, tag, view);
                if (input.screenshot) {
                    input.screenshot = false;
                    const VkExtent2D ext = presenter.extent();
                    RenderTarget shot = renderer.createTarget(ext.width, ext.height, true);
                    renderer.renderToMemory(shot, fi, rgba);
                    renderer.destroyTarget(shot);
                    const std::string dir = outputs.enabled() ? outputs.directory() : std::string(".");
                    std::filesystem::create_directories(dir);
                    const std::string path = dir + "/screenshot_" + std::to_string(sim.step()) + ".png";
                    if (writePNG(path, rgba.data(), static_cast<int>(ext.width), static_cast<int>(ext.height)))
                        std::printf("saved %s\n", path.c_str());
                }
                if (!presenter.present(fi)) glfwWaitEventsTimeout(0.05);

                if (since(lastTitle) > 0.5) {
                    const double rate = (sim.step() - stepsAtTitle) / since(lastTitle);
                    char title[256];
                    std::snprintf(title, sizeof(title), "FiniteCore - %s | t = %s s | %.0f steps/s | x%g%s",
                                  prob.title.c_str(), formatTime(sim.time(), sys.dt).c_str(), rate, input.speed,
                                  sys.diagonal ? "" : (" | CG " + std::to_string(sim.last.iterations) + " it").c_str());
                    glfwSetWindowTitle(window, title);
                    lastTitle = Clock::now();
                    stepsAtTitle = sim.step();
                }
                if (sim.finished() && opt.exitWhenDone) {
                    if (doneSince < 0) doneSince = 0.0;
                    doneSince += frameDt;
                    if (doneSince > 1.5) break;
                }
            }
        } else
#endif
        {
            auto lastPrint = Clock::now();
            bool progressShown = false;
            while (true) {
                if (sim.outputPending()) {
                    captureOutput();
                    continue;
                }
                if (sim.finished()) break;
                sim.advance(1 << 30);
                if (!opt.quiet && since(lastPrint) > 1.0) {
                    const double el = since(tRun);
                    std::printf("\r  t = %s / %g s (%3.0f%%)  %.0f steps/s", formatTime(sim.time(), sys.dt).c_str(),
                                sys.tEnd, 100.0 * sim.step() / sys.nSteps, sim.step() / el);
                    if (sim.implicitSteps > 0)
                        std::printf("  CG %.1f it/step", double(sim.totalIterations) / sim.implicitSteps);
                    std::printf("   ");
                    std::fflush(stdout);
                    lastPrint = Clock::now();
                    progressShown = true;
                }
            }
            if (progressShown) std::printf("\r%100s\r", "");
        }

        const double runSeconds = since(tRun);
        outputs.finish();
        if (outTarget.resolved.image) renderer.destroyTarget(outTarget);

        // ---- summary ------------------------------------------------------------------
        const int steps = sim.step();
        std::printf("done: %d steps to t = %g s in %.2f s wall (%.0f steps/s, %.3f ms/step on the GPU)\n", steps,
                    sim.time(), runSeconds, steps / std::max(runSeconds, 1e-9),
                    steps ? 1e3 * sim.gpuSeconds / steps : 0.0);
        if (sim.implicitSteps > 0) {
            std::printf("  CG iterations per step: avg %.1f, max %d", double(sim.totalIterations) / sim.implicitSteps,
                        sim.maxIterations);
            if (sim.unconvergedSteps) std::printf(", %d step(s) not converged", sim.unconvergedSteps);
            std::printf("\n");
        }
        if (opt.profile && gpu.profile.submits > 0)
            std::printf("  GPU profile: %lld submits, %lld dispatches; recording %.1f ms, submit+wait %.1f ms "
                        "(%.1f us per submit)\n",
                        gpu.profile.submits, gpu.profile.dispatches, 1e3 * gpu.profile.recordSeconds,
                        1e3 * gpu.profile.submitSeconds, 1e6 * gpu.profile.submitSeconds / gpu.profile.submits);
        std::printf("  final field: min %.6g  max %.6g  mean %.6g %s\n", stats.tmin, stats.tmax, stats.tmean,
                    prob.view.unit.c_str());
        if (outputs.hasExact())
            std::printf("  verification vs exact solution: max error %.4e (worst over time %.4e), rms %.4e\n",
                        outputs.lastErrorMax(), outputs.worstErrorMax(), outputs.lastErrorL2());
        if (reference) {
            std::printf("  CPU reference (fp64): max |T_gpu - T_cpu| = %.3e over all outputs\n", maxRefDiff);
            std::printf("  time stepping: GPU %.3f s vs CPU %.3f s  (%.1fx)\n", sim.gpuSeconds, sim.cpuSeconds,
                        sim.cpuSeconds / std::max(sim.gpuSeconds, 1e-12));
        }
        if (outputs.frames() > 0) {
            std::printf("  wrote %d frames to %s/:", outputs.frames(), outputs.directory().c_str());
            for (const std::string& f : outputs.writtenFiles()) std::printf(" %s", f.c_str());
            std::printf("\n");
        }
        if (sim.unconvergedSteps) exitCode = 2;
    }
#ifdef FC_HAVE_GLFW
    if (window) {
        glfwDestroyWindow(window);
        glfwTerminate();
    }
#endif
    return exitCode;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
