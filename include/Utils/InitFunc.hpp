#pragma once

namespace TDFANN {
namespace Utils {

inline void setup_logger(bool verbose, std::string name) {
    std::filesystem::create_directories("logs");

    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
        "logs/tdfann.log", true);

    console_sink->set_pattern("[%Y-%m-%d %H:%M:%S] [%^%l%$] %v");
    file_sink->set_pattern("[%Y-%m-%d %H:%M:%S] [%l] [%s:%#] %v");

    auto dist_sink = std::make_shared<spdlog::sinks::dist_sink_st>();
    dist_sink->add_sink(console_sink);
    dist_sink->add_sink(file_sink);

    auto logger = std::make_shared<spdlog::logger>(name, dist_sink);
    logger->set_level(spdlog::level::info);

    spdlog::set_default_logger(logger);

    if (verbose) {
        spdlog::set_level(spdlog::level::debug);
        spdlog::debug("Verbose logging enabled");
    }
}


}
}  // namespace TDFANN