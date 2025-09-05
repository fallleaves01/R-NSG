#pragma once

namespace TDFANN {
namespace Utils {

inline void setup_logger(bool verbose) {
    // 创建logs目录
    std::filesystem::create_directories("logs");

    // 创建多个sink（输出目标）
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
        "logs/tdfann.log", true);

    // 设置输出格式
    console_sink->set_pattern("[%Y-%m-%d %H:%M:%S] [%^%l%$] %v");
    file_sink->set_pattern("[%Y-%m-%d %H:%M:%S] [%l] [%s:%#] %v");

    // 创建分发sink，可以同时输出到多个目标
    auto dist_sink = std::make_shared<spdlog::sinks::dist_sink_st>();
    dist_sink->add_sink(console_sink);
    dist_sink->add_sink(file_sink);

    // 创建logger
    auto logger = std::make_shared<spdlog::logger>("tdfann", dist_sink);
    logger->set_level(spdlog::level::info);

    // 设置为默认logger
    spdlog::set_default_logger(logger);

    if (verbose) {
        spdlog::set_level(spdlog::level::debug);
        spdlog::debug("Verbose logging enabled");
    }
}


}
}  // namespace TDFANN